# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-19
from typing import List
from transformers import BertTokenizer


def find_lcs(query: List[str], attr_name: List[str]) -> List[int]:
    """找到query中每个token在attr_name中出现的位置"""
    query_len, attr_name_len = len(query), len(attr_name)
    # 用来保存对应位置的匹配结果
    match = [[0 for _ in range(attr_name_len + 1)] for _ in range(query_len + 1)]
    # 用来保存回溯的位置
    track = [[None for _ in range(attr_name_len + 1)] for _ in range(query_len + 1)]
    # 保存每个query中的token在attr_name中出现的位置
    occur_pos = [-1 for _ in range(query_len)]
    for row in range(query_len):
        for col in range(attr_name_len):
            if query[row] == attr_name[col]:
                # 字符匹配成功，该位置等于左上方的值+1
                match[row + 1][col + 1] = match[row][col] + 1
                track[row + 1][col + 1] = "ok"
            elif match[row + 1][col] >= match[row][col + 1]:
                # 左值大于上值，则该位置的值为左值转移而来，标记回溯方向为左
                match[row + 1][col + 1] = match[row + 1][col]
                track[row + 1][col + 1] = "left"
            else:
                # 上值大于左值，则该位置的值为上值转移而来，标记回溯方向为上
                match[row + 1][col + 1] = match[row][col + 1]
                track[row + 1][col + 1] = "up"
    # print("match: ", match)
    # print("track: ", track)
    # 从后向前回溯
    row, col = query_len, attr_name_len
    while match[row][col]:
        # 获取回溯位置的标记
        tag = track[row][col]
        # 如果匹配成功，记录该字符在query中对应的attr_name中的位置
        if tag == "ok":
            # 向前找匹配并且是ok的位置，而且match值不能小于当前位置
            origin_col = col + 1
            cur_col = col - 1
            for k in range(col - 1, 0, -1):
                if match[row][k] < match[row][col]:
                    cur_col = k
                    break
            cur_col += 1
            # 向后找第一个ok的位置，保证词尽可能连续
            first_col = origin_col - 1
            for k in range(cur_col, origin_col):
                if track[row][k] == "ok":
                    first_col = k
                    break
            last_col = origin_col - 1
            cur_len = match[row][first_col]
            if first_col != last_col:
                # 如果当前token前面没有token了，就不用取前面的了
                if cur_len <= 1:
                    col = last_col
                else:
                    col = first_col
            else:
                col = first_col
            row -= 1
            col -= 1
            occur_pos[row] = col
        # 向左边找上一个匹配的位置
        elif tag == "left":
            col -= 1
        # 向上面找上一个匹配的位置
        elif tag == "up":
            row -= 1

    return occur_pos


def find_substring_pos_pair(query: List[str],
                            occur_pos: List[int],
                            tokenizer: BertTokenizer):
    """
    根据query中单词在attr_name中出现的位置
    找到最长匹配的子串
    以及最后一个匹配单词对应的位置
    """
    occur_pos_len = len(occur_pos)
    max_len = 0
    query_attr_pos_pair = [-1, -1]

    cur_pos = 0
    while cur_pos < occur_pos_len:
        # 如果当前query位置中token在attr_name中出现过
        if occur_pos[cur_pos] != -1:
            substring_first_pos = cur_pos
            while (cur_pos < occur_pos_len) and (occur_pos[cur_pos] != -1):
                cur_pos += 1

            substring_last_pos = cur_pos - 1
            # 计算当前substring的长度
            cur_len = substring_last_pos - substring_first_pos + 1
            # 允许存在连续的UNK token，但是占比必须小于一半
            if ((cur_len >= 2) and (cur_len > max_len) and
                    (query[substring_first_pos:substring_last_pos+1].count(tokenizer.unk_token) / cur_len < 0.5)):
                max_len = cur_len
                query_attr_pos_pair = [substring_last_pos, occur_pos[substring_last_pos]]
        else:
            cur_pos += 1

    return query_attr_pos_pair