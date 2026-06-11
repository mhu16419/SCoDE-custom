import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .constants import CWE_DESCRIPTIONS, LANGUAGE_MAPS
from .utils import try_parse


DEFAULT_REWARD_WEIGHTS = {
    'parse': 1.0,
    'security': 1.0,
    'codeql': 2.0,
    'tests': 2.0,
}


@dataclass
class RewardScore:
    total: float
    parse: Optional[float] = None
    security: Optional[float] = None
    codeql: Optional[float] = None
    tests: Optional[float] = None
    codeql_warnings: Optional[int] = None
    test_status: Optional[str] = None
    critique: str = ''

    def to_dict(self):
        return asdict(self)


def reward_guided_enabled(args) -> bool:
    return bool(getattr(args, 'reward_guided', False))


def candidate_budget(args) -> int:
    multiplier = max(1, int(getattr(args, 'candidate_multiplier', 1)))
    return int(args.num_samples) * multiplier


def parse_reward_weights(raw_weights) -> dict:
    if raw_weights is None or raw_weights == '':
        return dict(DEFAULT_REWARD_WEIGHTS)
    if isinstance(raw_weights, dict):
        weights = dict(DEFAULT_REWARD_WEIGHTS)
        weights.update({k: float(v) for k, v in raw_weights.items()})
        return weights

    weights = dict(DEFAULT_REWARD_WEIGHTS)
    for item in raw_weights.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' in item:
            key, value = item.split('=', 1)
        elif ':' in item:
            key, value = item.split(':', 1)
        else:
            raise ValueError(f'Invalid reward weight "{item}". Use key=value.')
        key = key.strip()
        if key not in DEFAULT_REWARD_WEIGHTS:
            raise ValueError(f'Unknown reward component "{key}".')
        weights[key] = float(value)
    return weights


def format_reward_input(code: str, info: Optional[dict]) -> str:
    info = info or {}
    lang = info.get('language', 'unknown')
    language = LANGUAGE_MAPS.get(lang, lang)
    cwe = info.get('cwe') or info.get('vul_type') or ''
    cwe_desc = CWE_DESCRIPTIONS.get(str(cwe).upper(), '')
    description = info.get('description', '')

    header = [f'Language: {language}']
    if cwe:
        header.append(f'CWE: {str(cwe).upper()}')
    if cwe_desc:
        header.append(f'CWE description: {cwe_desc}')
    if description:
        header.append(f'Task: {description}')
    return '\n'.join(header) + '\n\nCode:\n' + code


def build_repair_prompt(prompt: str, critiques: Iterable[str]) -> str:
    critiques = [c for c in critiques if c]
    if len(critiques) == 0:
        return prompt
    lessons = '\n'.join(f'- {critique}' for critique in critiques)
    return (
        'Previous generated attempts had these issues:\n'
        f'{lessons}\n'
        'Generate a corrected completion that avoids these issues.\n\n'
        f'{prompt}'
    )


def write_reward_records(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


class RewardScorer:
    def __init__(self, args=None, model_path=None, weights=None, device=None, max_length=None):
        self.args = args
        self.model_path = model_path if model_path is not None else getattr(args, 'reward_model_path', None)
        self.weights = parse_reward_weights(weights if weights is not None else getattr(args, 'reward_weights', None))
        self.max_length = int(max_length if max_length is not None else getattr(args, 'reward_max_length', 512))
        self.tokenizer = None
        self.model = None

        if device is None:
            device = getattr(args, 'device', None)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)

        if self.model_path:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f'Reward model path does not exist: {self.model_path}')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.to(self.device)
            self.model.eval()

    def score_sources(
        self,
        sources: List[str],
        info: Optional[dict],
        parse_results: Optional[List[Optional[bool]]] = None,
        codeql_warnings: Optional[List[Optional[int]]] = None,
        test_results: Optional[List[Optional[object]]] = None,
    ) -> List[RewardScore]:
        if parse_results is None:
            parse_results = [None] * len(sources)
        if codeql_warnings is None:
            codeql_warnings = [None] * len(sources)
        if test_results is None:
            test_results = [None] * len(sources)

        security_scores = self._security_scores(sources, info)
        scores = []
        for source, parse_ok, warnings, test_result, security_score in zip(
            sources, parse_results, codeql_warnings, test_results, security_scores
        ):
            parse_score = self._parse_score(source, info, parse_ok)
            codeql_score = None if warnings is None else 1.0 / (1.0 + max(0, int(warnings)))
            tests_score, test_status = self._test_score(test_result)
            total = self._weighted_total(
                parse=parse_score,
                security=security_score,
                codeql=codeql_score,
                tests=tests_score,
            )
            critique = self._critique(parse_score, security_score, warnings, tests_score, test_status)
            scores.append(RewardScore(
                total=total,
                parse=parse_score,
                security=security_score,
                codeql=codeql_score,
                tests=tests_score,
                codeql_warnings=warnings,
                test_status=test_status,
                critique=critique,
            ))
        return scores

    def select_top_indices(self, scores: List[RewardScore], limit: int) -> List[int]:
        return sorted(range(len(scores)), key=lambda idx: scores[idx].total, reverse=True)[:limit]

    def build_repair_memory(self, scores: List[RewardScore], limit: int) -> List[str]:
        critiques = []
        seen = set()
        for score in sorted(scores, key=lambda item: item.total):
            if not score.critique or score.critique in seen:
                continue
            critiques.append(score.critique)
            seen.add(score.critique)
            if len(critiques) >= limit:
                break
        return critiques

    def _security_scores(self, sources: List[str], info: Optional[dict]) -> List[Optional[float]]:
        if self.model is None:
            return [None] * len(sources)
        if len(sources) == 0:
            return []

        inputs = [format_reward_input(source, info) for source in sources]
        scores = []
        batch_size = int(getattr(self.args, 'reward_batch_size', 8))
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = inputs[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                ).to(self.device)
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
                if probs.size(-1) == 1:
                    batch_scores = torch.sigmoid(logits[:, 0])
                else:
                    batch_scores = probs[:, 1]
                scores.extend(batch_scores.detach().cpu().tolist())
        return [float(score) for score in scores]

    def _parse_score(self, source: str, info: Optional[dict], parse_ok: Optional[bool]) -> Optional[float]:
        if info is None:
            return None
        if parse_ok is None:
            try:
                parse_ok = try_parse(source, info) == 0
            except Exception:
                parse_ok = False
        return 1.0 if parse_ok else 0.0

    def _test_score(self, test_result: Optional[object]):
        if test_result is None:
            return None, None
        if isinstance(test_result, bool):
            return (1.0 if test_result else 0.0), ('OK' if test_result else 'FAIL')
        if isinstance(test_result, dict):
            status = test_result.get('status')
        else:
            status = str(test_result)
        return (1.0 if status == 'OK' else 0.0), status

    def _weighted_total(self, **signals) -> float:
        numerator = 0.0
        denominator = 0.0
        for name, value in signals.items():
            if value is None:
                continue
            weight = self.weights.get(name, 0.0)
            numerator += weight * float(value)
            denominator += weight
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _critique(
        self,
        parse_score: Optional[float],
        security_score: Optional[float],
        codeql_warnings: Optional[int],
        tests_score: Optional[float],
        test_status: Optional[str],
    ) -> str:
        if parse_score == 0.0:
            return 'The candidate did not parse or compile; preserve syntax and close all scopes.'
        if codeql_warnings is not None and codeql_warnings > 0:
            return 'Static analysis reported a security warning; avoid unsafe data flow and vulnerable APIs.'
        if tests_score == 0.0:
            return f'Unit tests failed with status {test_status}; preserve the required behavior and edge cases.'
        if security_score is not None and security_score < 0.5:
            return 'The learned security reward judged the code risky; prefer validation, escaping, bounds checks, and safe APIs.'
        return 'Keep the candidate syntactically valid, behaviorally correct, and secure.'
