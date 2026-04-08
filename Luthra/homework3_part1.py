import numpy as np

c = 20
R = 8.314
T = 800
P = 101.325
kfwd = 2.10e13
krev = 1.81e10

class compound:
    def __init__(self,partgo,vfwd,vrev,x):
        self.partgo = partgo
        self.vfwd = vfwd
        self.vrev = vrev
        self.x = x
        self.c = x*c
        self.partg = partgo + R*T/1000*np.log(self.c)
        self.partgrxn = (vrev - vfwd)*self.partg
        self.fwdprod = self.c**vfwd
        self.revprod = self.c**vrev


ch4_1 = compound(-236.14,1,0,0.147)
o2_1 = compound(-172.93,2,0,0.324)
h2o_1 = compound(-402.89,0,2,0.460)
co2_1 = compound(-576.71,0,1,0.069)

ch4_2 = compound(-236.14,1,0,0.147)
o2_2 = compound(-172.93,2,0,0.206)
h2o_2 = compound(-402.89,0,2,0.579)
co2_2 = compound(-576.71,0,1,0.069)

ch4_3 = compound(-236.14,1,0,0.005)
o2_3 = compound(-172.93,2,0,0.260)
h2o_3 = compound(-402.89,0,2,0.396)
co2_3 = compound(-576.71,0,1,0.339)

grxn1 = ch4_1.partgrxn + o2_1.partgrxn + h2o_1.partgrxn + co2_1.partgrxn
grxn2 = ch4_2.partgrxn + o2_2.partgrxn + h2o_2.partgrxn + co2_2.partgrxn
grxn3 = ch4_3.partgrxn + o2_3.partgrxn + h2o_3.partgrxn + co2_3.partgrxn

q1 = kfwd*ch4_1.fwdprod*o2_1.fwdprod*h2o_1.fwdprod*co2_1.fwdprod - krev*ch4_1.revprod*o2_1.revprod*h2o_1.revprod*co2_1.revprod
q2 = kfwd*ch4_2.fwdprod*o2_2.fwdprod*h2o_2.fwdprod*co2_2.fwdprod - krev*ch4_2.revprod*o2_2.revprod*h2o_2.revprod*co2_2.revprod
q3 = kfwd*ch4_3.fwdprod*o2_3.fwdprod*h2o_3.fwdprod*co2_3.fwdprod - krev*ch4_3.revprod*o2_3.revprod*h2o_3.revprod*co2_3.revprod

print(grxn1)
print(grxn2)
print(grxn3)
print(f"{q1:e}")
print(f"{q2:e}")
print(f"{q3:e}")



# compositions = np.array([[0.147,0.147,0.005],[0.324,0.206,0.260],[0.460,0.069,0.396],[0.069,0.069,0.339]])
# grxn = np.empty(3,)
# q = np.empty(3,)

# for case in range(2):
#     ch4 = compound(-236.14,1,0,compositions[0,case])
#     o2 = compound(-172.93,2,0,compositions[1,case])
#     h2o = compound(-402.89,0,2,compositions[2,case])
#     co2 = compound(-576.71,0,1,compositions[3,case])
#     grxn[case,] = ch4.partgrxn + o2.partgrxn + h2o.partgrxn + co2.partgrxn
#     q[case,] = kfwd*ch4.fwdprod*o2.fwdprod*h2o.fwdprod*co2.fwdprod - krev*ch4.revprod*o2.revprod*h2o.revprod*co2.revprod

# print(grxn)
# print(f"{q:e}")