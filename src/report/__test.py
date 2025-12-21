filter_1 = [1,2,3]

filter_2 = [4,5,6]

filter_3 = [7,8,9,10]

filter_list = [filter_1, filter_2, filter_3]

result = None

for i in filter_list:
    result = [
        [k for k in j] for j in i
    ]

print(result)