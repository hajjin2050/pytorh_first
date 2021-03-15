#텐서는 파이토치의 가장 기본이 되는 데이터 구조
#텐서는 각 데이터형별로 정의돼 있음
#32비트의 유동소수점은 torch.FloatTensor
#64비트의 부호있는 정수라면 torch.LongTensor

import numpy as np
import torch

#중첩 list를 지정
t = torch.tensor([[1,2],[3,4.]])

#device를 지정하면 GPU에 텐서를 만들수있음
t = torch.tensor([[1,2],[3,4.]],device="cuda:0")

#dtype을 사용해 데이터형을 지정하여 텐서를 만들 수 있다.
t = torch.tensor([[1,2],[3,4.]],dtype = torch.float64)

#0부터 0까지의 수치로 초기화로 1차원 텐서
t = torch.arange(0, 10)

#모든 값이 0인 100 x 10의 텐서를
#작성해서 to 메서드로 GPU에 전송
t = torch.zeros(100, 10).to("cuda:0")

#정규 난수로 100 x 10 의 텐서를 작성
t = torch.randn(100, 10)

#텐서의 shape은 size 메서드로 확인 가능
t.size()

print(t.size())