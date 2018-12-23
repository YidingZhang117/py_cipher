class MyTime():
    def __init__(self,hour=0,minute=0,second=0, kaixin=100):
        self.hour = hour
        self.minute = minute
        self.second = second
        self.zyd2clj(kaixin)

    def zyd2clj(self,kaixin=100):
        self.kaixin = kaixin

    def __str__(self):
        return '%.2d:%.2d:%.2d---%.2f' % (self.hour,self.minute,self.second,self.kaixin)

t1 = MyTime(2,3,kaixin=3)
# t1.zyd2clj(3)
print(t1)