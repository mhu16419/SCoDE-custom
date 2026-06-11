import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from safecoder.reward_dataset import RewardDataset
from safecoder.utils import set_logging, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', default=['sec-desc', 'sec-new-desc'])
    parser.add_argument('--base_model', type=str, default='microsoft/codebert-base')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_num_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_acc_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--model_dir', type=str, default='../trained')
    args = parser.parse_args()

    output_name = args.output_name
    if not output_name.endswith('-reward'):
        output_name = f'{output_name}-reward'
    args.output_dir = os.path.join(args.model_dir, output_name)
    return args


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch['labels']
            correct += (preds == labels).sum().item()
            total += labels.numel()
    model.train()
    avg_loss = total_loss / max(1, len(dataloader))
    acc = correct / max(1, total)
    return avg_loss, acc


def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train_reward.log'))
    set_seed(args.seed)

    device_name = args.device
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        device_name = 'cpu'
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    train_dataset = RewardDataset(args, tokenizer, 'train')
    val_dataset = RewardDataset(args, tokenizer, 'val')
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        drop_last=False,
    )

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    total_steps = max(1, len(train_loader) // max(1, args.grad_acc_steps) * args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    args.logger.info(f'Training reward model with args: {args}')
    args.logger.info('  Num train samples = %d', len(train_dataset))
    args.logger.info('  Num val samples = %d', len(val_dataset))
    args.logger.info('  Batch size = %d', args.batch_size)
    args.logger.info('  Gradient accumulation steps = %d', args.grad_acc_steps)
    args.logger.info('  Total optimization steps = %d', total_steps)

    global_step = 0
    model.train()
    optimizer.zero_grad()
    for epoch in range(args.num_train_epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / max(1, args.grad_acc_steps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            running_loss += loss.item()

            if (step + 1) % args.grad_acc_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_train_loss = running_loss / max(1, args.logging_steps)
                    val_loss, val_acc = evaluate(model, val_loader, device)
                    args.logger.info(
                        'step=%d train_loss=%.6f val_loss=%.6f val_acc=%.4f',
                        global_step,
                        avg_train_loss,
                        val_loss,
                        val_acc,
                    )
                    running_loss = 0.0

        val_loss, val_acc = evaluate(model, val_loader, device)
        args.logger.info('epoch=%d val_loss=%.6f val_acc=%.4f', epoch + 1, val_loss, val_acc)
        if (epoch + 1) % args.save_epochs == 0:
            save_model(model, tokenizer, os.path.join(args.output_dir, f'checkpoint-epoch-{epoch + 1}'))

    save_model(model, tokenizer, args.output_dir)


if __name__ == '__main__':
    main()
