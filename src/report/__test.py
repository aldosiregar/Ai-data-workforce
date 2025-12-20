from numpy import array

test = [
    [
        [1,2],
        [3,4]
    ],
    [
        [5,6],
        [7,8]
    ],
    [
        [9,10],
        [11,12]
    ]
]

class testing:
    index = 1
    temp = []

    def crawler(self,x=[]):
        for i in x:
            if(type(i) == list):
                self.crawler(x=i)
            else:
                print(i)
                self.temp.append(i)

obj = testing()

obj.crawler(test)

print(obj.temp)