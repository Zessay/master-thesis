# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-13
from transformers import BertTokenizer


vocab_file = r"E:\Models\CDial-GPT2_LCCC-base\vocab.txt"
do_lower_case = True

tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

print(tokenizer.mask_token_id)

# print(tokenizer.all_special_tokens)
#
# output_ids = [10, 192, 288, 200, 873, 882]
#
# print(tokenizer.decode(output_ids, skip_special_tokens=True))