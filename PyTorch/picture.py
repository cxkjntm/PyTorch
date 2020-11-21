#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2020/11/14 20:41
# @Author  : lxy

import matplotlib.pyplot as plt
import pandas as pd

file_path = 'C:\\Users\\lxy\\Desktop\\data.xlsx'
df = pd.read_excel(file_path,header=0)
df.columns.reindex(range(1,30))
print(df)
plt.plot(df.index,df['TFIDF-LDA'],marker='o',mec='r',mfc='w', label="TFIDF-LDA")
plt.plot(df.index,df['SLDA-LDA'],marker='*',ms=10, label="SLDA-LDA")
plt.plot(df.index,df['co-SLDA'],marker='s',ms=5, label="co-sLDA")
plt.xlabel("iteration times")
plt.ylabel("f1(%)")
plt.legend()
plt.show()
