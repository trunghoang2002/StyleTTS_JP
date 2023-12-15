# IPA Phonemizer: https://github.com/bootphon/phonemizer

# _letters_ipa = _letters_ipa = """<blank>
# <unk>
# a
# o
# i
# u
# e
# k
# n
# t
# r
# s
# N
# m
# pau
# sh
# d
# g
# w
# U
# I
# cl
# h
# y
# b
# j
# ts
# ch
# z
# p
# f
# ky
# ry
# gy
# hy
# ny
# by
# my
# py
# v
# dy
# ty
# <sos/eos>"""

# # Export all symbols:
# symbols = _letters_ipa.split('\n')

# dicts = {}
# for i in range(len((symbols))):
#     dicts[symbols[i]] = i

# class TextCleaner:
#     def __init__(self, dummy=None):
#         self.word_index_dictionary = dicts
#         print(len(dicts))
#     def __call__(self, text):
#         indexes = []
#         for char in text:
#             try:
#                 indexes.append(self.word_index_dictionary[char])
#             except KeyError:
#                 print(text)
#         return indexes
