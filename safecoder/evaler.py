import os
import re
import abc
import numpy as np
import torch
import torch.nn as nn

try:
    import openai
except ImportError:
    openai = None


from .utils import load_model, set_seed, try_parse
from .constants import PROMPT_NO_INPUT, INSTRUCTION, LANGUAGE_MAPS, PRETRAINED_MODELS
from .constants import SECURE_PROMPTING_GENERIC, SECURE_PROMPTING_SPECIFIC, CWE_DESCRIPTIONS
from .constants import SECURE_PROMPTING, INSECURE_PROMPTING
from .reward import RewardScorer, build_repair_prompt, candidate_budget, reward_guided_enabled
from steer.steer_models import Steer

import time


def truncate_after(completion, trunc_str):
    return completion[:completion.find(trunc_str) + len(trunc_str)]


def truncate_before(completion, trunc_str):
    return completion[:completion.find(trunc_str)].rstrip()


def truncate_after_last(completion, trunc_str):
    return completion[:completion.rfind(trunc_str) + len(trunc_str)]


def truncate_before_last(completion, trunc_str):
    return completion[:completion.rfind(trunc_str)]


class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.model = load_model(args.model_name, args)
        self.reward_scorer = None

    def _get_reward_scorer(self):
        if getattr(self, 'reward_scorer', None) is None:
            self.reward_scorer = RewardScorer(self.args)
        return self.reward_scorer

    def _generation_budget(self):
        if reward_guided_enabled(self.args):
            return candidate_budget(self.args)
        return self.args.num_samples

    def _generation_rounds(self):
        if not reward_guided_enabled(self.args):
            return 1
        return max(0, int(getattr(self.args, 'repair_rounds', 0))) + 1

    def _num_batches(self, budget):
        per_gen = max(1, int(self.args.num_samples_per_gen))
        return max(1, (budget + per_gen - 1) // per_gen)

    def _repair_memory(self, sources, info):
        if len(sources) == 0:
            return []
        scorer = self._get_reward_scorer()
        scores = scorer.score_sources(sources, info)
        return scorer.build_repair_memory(scores, int(getattr(self.args, 'repair_memory_size', 3)))

    def _finalize_sources(self, sources, info):
        if reward_guided_enabled(self.args) and not getattr(self.args, 'reward_defer_selection', False):
            scorer = self._get_reward_scorer()
            scores = scorer.score_sources(sources, info)
            selected = scorer.select_top_indices(scores, self.args.num_samples)
            sources = [sources[idx] for idx in selected]

        output_srcs, non_parsed_srcs = [], []
        for src in sources:
            if info['language'] != 'go' and try_parse(src, info) != 0:
                non_parsed_srcs.append(src)
            else:
                output_srcs.append(src)
        return output_srcs, non_parsed_srcs

    def sample(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        output_candidates, memory = [], []

        for repair_round in range(self._generation_rounds()):
            cur_prompt = build_repair_prompt(prompt, memory) if memory else prompt
            input_ids = self.tokenizer.encode(cur_prompt, return_tensors='pt').to(self.model.device)
            input_ids_len = input_ids.size(1)

            for i in range(self._num_batches(self._generation_budget())):
                set_seed(self.args.seed + repair_round * 1000 + i)

                gen_output = self.model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=self.args.num_samples_per_gen,
                    temperature=self.args.temp,
                    max_new_tokens=self.args.max_gen_len,
                    top_p=self.args.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

                tokens = gen_output[:, input_ids_len:, ...]
                completions = self.tokenizer.batch_decode(tokens)
                for completion in completions:
                    if self.tokenizer.eos_token in completion:
                        completion = completion[:completion.find(self.tokenizer.eos_token)]
                    completion = self.postprocess(completion, info)
                    output_src = file_context + func_context + completion
                    output_candidates.append(output_src.rstrip() + '\n')

            if repair_round + 1 < self._generation_rounds():
                memory = self._repair_memory(output_candidates, info)

        return self._finalize_sources(output_candidates, info)

    @abc.abstractclassmethod
    def preprocess(self, file_context, func_context, info):
        raise NotImplementedError()

    def postprocess(self, completion, info):
        if info['language'] == 'py':
            for match in re.finditer('\n', completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                if '\n    #' in completion:
                    completion = truncate_before_last(completion, '\n    #')
        elif info['language'] in ['c', 'cpp']:
            if '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif ';\n' in completion:
                completion = truncate_after_last(completion, ';\n') + '\n}'
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'go':
            if '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'js':
            if '\n});' in completion: # for app function definitions
                completion = truncate_after(completion, '\n});')
            elif re.search(r'\n}(?!;)', completion) is not None: # normal function end
                match = re.search(r'\n}(?!;)', completion)
                completion = completion[:match.end()]
            elif '\n//' in completion:
                completion = truncate_before_last(completion, '\n//').rstrip()
            elif '\n/*' in completion:
                completion = truncate_before_last(completion, '\n/*').rstrip()
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'jsx':
            # only for cwe-200 0-jsx
            if '\n' in completion:
                completion = truncate_before(completion, '\n')
        elif info['language'] == 'rb':
            if '\n    end' in completion:
                completion = truncate_after(completion, '\n    end') + '\nend'
            elif '\nend' in completion:
                completion = truncate_after(completion, '\nend')
            elif '    #' in completion:
                completion = truncate_before_last(completion, '    #').rstrip('\n') + '\nend'
                if '\nend' not in completion: completion += '\nend'
            else:
                completion = completion
        elif info['language'] == 'java':
            if '\n    }' in completion:
                completion = truncate_after(completion, '\n    }') + '\n}'
            elif '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif ';\n' in completion:
                completion = truncate_after_last(completion, ';\n') + '\n    }' + '\n}'
            elif '    //' in completion:
                completion = truncate_before_last(completion, '    //').rstrip('\n') + '\n}'
                if '\n}' not in completion: completion += '\n}'
            elif '    /*' in completion:
                completion = truncate_before_last(completion, '    /*').rstrip('\n') + '\n}'
                if '\n}' not in completion: completion += '\n}'
            else:
                completion = completion
        else:
            raise NotImplementedError('Postprocessing for {language} is not implemented yet'.format(language=info['language']))

        if 'postprocess' in info:
            scope = {'completion': completion}
            exec(info['postprocess'], scope)
            completion = scope['completion']

        return completion

class EvalerCodePLM(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context

class EvalerCodeFT(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        lang = LANGUAGE_MAPS[info['language']]
        if self.args.sec_prompting == 'generic':
            instruction = SECURE_PROMPTING_GENERIC.format_map({'language': lang, 'prompt': info['description']})
        elif self.args.sec_prompting == 'specific':
            instruction = SECURE_PROMPTING_SPECIFIC.format_map({'language': lang, 'prompt': info['description'], 'cwe': info['cwe'], 'cwe_desc': CWE_DESCRIPTIONS[info['cwe']]})
        else:
            instruction = INSTRUCTION.format_map({'language': lang, 'prompt': info['description']})
        prompt = PROMPT_NO_INPUT.format_map({'instruction': instruction})
        prompt += file_context + func_context
        return prompt


class EvalerChat(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        lang = LANGUAGE_MAPS[info['language']]

        if self.args.sec_prompting == 'generic':
            instruction = SECURE_PROMPTING_GENERIC.format_map({'language': lang, 'prompt': info['description']})
        elif self.args.sec_prompting == 'specific':
            instruction = SECURE_PROMPTING_SPECIFIC.format_map({'language': lang, 'prompt': info['description'], 'cwe': info['cwe'], 'cwe_desc': CWE_DESCRIPTIONS[info['cwe']]})
        else:
            instruction = INSTRUCTION.format_map({'language': lang, 'prompt': info['description']})

        if self.args.model_name == 'octocoder':
            template = 'Question: {instruction}\n\nAnswer: \n'
            prompt = template.format_map({'instruction': instruction})
            prompt += file_context + func_context
        else:
            if self.args.model_name == 'deepseek':
                prompt = instruction
            else:
                prompt = PROMPT_NO_INPUT[:PROMPT_NO_INPUT.rfind('\n\n')].format_map({'instruction': instruction})
            messages = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': file_context+func_context}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            prompt = prompt.removeprefix('<s>').removesuffix('</s> ').removesuffix(' </s>').removesuffix('\n<|EOT|>\n')
        return prompt

class EvalerOpenAI(EvalerBase):
    def __init__(self, args):
        if openai is None:
            raise ImportError('The openai package is required for EvalerOpenAI.')
        self.args = args
        self.model = args.model_name
        self.client = openai.OpenAI()

    def _extract_markdown(self, md):
        pattern = r'```.*?\n(.*?)```'
        matches = re.findall(pattern, md, re.DOTALL)
        return matches

    def sample(self, file_context, func_context, info):
        lang = info['language']

        if self.args.sec_prompting == 'generic':
            instruction = SECURE_PROMPTING_GENERIC.format_map({'language': lang, 'prompt': info['description']})
        elif self.args.sec_prompting == 'specific':
            instruction = SECURE_PROMPTING_SPECIFIC.format_map({'language': lang, 'prompt': info['description'], 'cwe': info['cwe'], 'cwe_desc': CWE_DESCRIPTIONS[info['cwe']]})
        else:
            instruction = INSTRUCTION.format_map({'language': lang, 'prompt': info['description']})
        prompt = PROMPT_NO_INPUT.format_map({'instruction': instruction})
        prompt += file_context+func_context

        srcs, memory = [], []
        for repair_round in range(self._generation_rounds()):
            cur_prompt = build_repair_prompt(prompt, memory) if memory else prompt
            for i in range(self._num_batches(self._generation_budget())):
                response = self.client.completions.create(
                    model=self.model,
                    prompt=cur_prompt,
                    n=self.args.num_samples_per_gen,
                    temperature=self.args.temp,
                    max_tokens=self.args.max_gen_len,
                    top_p=self.args.top_p,
                    seed=self.args.seed + repair_round * 1000 + i
                )
                for choice in response.choices:
                    completion = choice.text
                    completion = self.postprocess(completion, info)
                    srcs.append((file_context + func_context + completion).rstrip() + '\n')

            if repair_round + 1 < self._generation_rounds():
                memory = self._repair_memory(srcs, info)

        return self._finalize_sources(srcs, info)


class EvalerCodeSTEER(EvalerCodeFT):
    def __init__(self, args):
        self.args = args
        self.model = Steer(args, args.model_name, args.num_steers, args.rank, args.epsilon, args.init_var)
        self.tokenizer = self.model.tokenizer
        ckpt_name = os.path.join(args.model_dir, args.model_name, 'checkpoint-last', 'pytorch_model.bin')
        self.model.load_state_dict(torch.load(ckpt_name, map_location=self.model.device))

        if args.wo_st:
            embed_dim = self.model.get_embed_dim()
            # random initialization
            projector1 = nn.Parameter(torch.randn(
                args.num_steers, embed_dim, args.rank
            ) * args.init_var).to(self.model.device)
            projector2 = nn.Parameter(torch.randn(
                args.num_steers, embed_dim, args.rank
            ) * args.init_var).to(self.model.device)
            self.model.model.lm_head.projector1.data = projector1.data
            self.model.model.lm_head.projector2.data = projector2.data


    def sample(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        output_candidates, memory = [], []

        for repair_round in range(self._generation_rounds()):
            cur_prompt = build_repair_prompt(prompt, memory) if memory else prompt
            input_ids = self.tokenizer.encode(cur_prompt, return_tensors='pt').to(self.model.device)
            input_ids_len = input_ids.size(1)

            for i in range(self._num_batches(self._generation_budget())):
                set_seed(self.args.seed + repair_round * 1000 + i)

                gen_output = self.model.generate(
                    input_ids,
                    steer_values=list(map(float, self.args.steer_values)) if self.args.steer_values is not None else None,
                    do_sample=True,
                    num_return_sequences=self.args.num_samples_per_gen,
                    temperature=self.args.temp,
                    max_new_tokens=self.args.max_gen_len,
                    top_p=self.args.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

                tokens = gen_output[:, input_ids_len:, ...]
                completions = self.tokenizer.batch_decode(tokens)
                for completion in completions:
                    if self.tokenizer.eos_token in completion:
                        completion = completion[:completion.find(self.tokenizer.eos_token)]
                    completion = self.postprocess(completion, info)
                    output_src = file_context + func_context + completion
                    output_candidates.append(output_src.rstrip() + '\n')

            if repair_round + 1 < self._generation_rounds():
                memory = self._repair_memory(output_candidates, info)

        return self._finalize_sources(output_candidates, info)


class EvalerCodeCOSEC(EvalerCodeFT):
    def __init__(self, args):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cosec.CustomizedGeneration import CodeLlamaModelLM, StarcodeModelLM, CodegenModelLM, Qwen2ModelLM
        self.args = args
        if 'deepseek' in args.model_name_or_path:
            self.model = CodeLlamaModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
            self.sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
        elif 'codellama' in args.model_name_or_path:
            self.model = CodeLlamaModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
            self.sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
        elif 'qwen2.5' in args.model_name_or_path:
            self.model = Qwen2ModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
            self.sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
        elif 'star' in args.model_name_or_path:
            self.model = StarcodeModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
            self.sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
        elif 'codegen' in args.model_name_or_path:
            self.model = CodegenModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
            self.sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
        else:
            raise NotImplementedError()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.model.eval()
        self.sec_model.eval()

    def sample(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        output_candidates, memory = [], []

        for repair_round in range(self._generation_rounds()):
            cur_prompt = build_repair_prompt(prompt, memory) if memory else prompt
            input_ids = self.tokenizer.encode(cur_prompt, return_tensors='pt').to(self.model.device)
            input_ids_len = input_ids.size(1)

            for i in range(self._num_batches(self._generation_budget())):
                set_seed(self.args.seed + repair_round * 1000 + i)
                kwargs = {
                    'expert': True,
                    'expert_lm': self.sec_model,
                    'model_kwargs_expert': {},
                    'threshold': self.args.threshold,
                }

                gen_output = self.model.generate_with_experts(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=self.args.num_samples_per_gen,
                    temperature=self.args.temp,
                    max_new_tokens=self.args.max_gen_len,
                    top_p=self.args.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    expert_min_prob=0.0,
                    expert_temperature=self.args.exp_temp,
                    expert_top_p=0.95,
                    **kwargs
                )

                tokens = gen_output[:, input_ids_len:, ...]
                completions = self.tokenizer.batch_decode(tokens)
                for completion in completions:
                    if self.tokenizer.eos_token in completion:
                        completion = completion[:completion.find(self.tokenizer.eos_token)]
                    completion = self.postprocess(completion, info)
                    output_src = file_context + func_context + completion
                    output_candidates.append(output_src.rstrip() + '\n')

            if repair_round + 1 < self._generation_rounds():
                memory = self._repair_memory(output_candidates, info)

        return self._finalize_sources(output_candidates, info)
