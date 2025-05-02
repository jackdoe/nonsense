```
def _unitary(self, h_real):
    H = 0.5*(A + A^T)                # make a symmetric real matrix
    U = expm(−i·H·dt)               # a learnable unitary via matrix exponential
    return h_c + (h_c @ U)
h_c = self._unitary(self.ln2(h_c.real).to(torch.cfloat))
```

experimenting with unitary feature mixer `U = e^{−iHΔt}` 



```
step     0 | train 222.589 | val 221.980 | 0.0 min
»                                                                                                                                                                                                           






step  1000 | train 2.499 | val 2.585 | 14.2 min
»  bea f age-
Titagups se p irare i was t te llin ashe ineas to idieac a d ill eipea t ieaim
Ain a mas y heke
Tons d at ill s a hitea f ate f I hoetl t tha dis m im y
Yoite heais, t ce ais d andis tit, m 

step  2000 | train 1.827 | val 1.986 | 28.0 min
»  to make in live a this lis,
Or leass feor that Give and rices,
And live mane any, and you ad gargaty, s mpak's fand ther.

ROMEO:
Do this for app darence fanthis: s be!
My gring than nmere me man han  


step  3000 | train 1.497 | val 1.766 | 42.7 min
»  be commend I saw that be.

ELBOW:
O what is comes! what descept the all thou supppers be to the dus.

PAGE:
I the know ther ithe kin the is mosthe the king made ffrom thes maf We man he suppre was s t 

step  4000 | train 1.375 | val 1.687 | 57.9 min
»  commitakes.

ESCALUS:
Ay, then them be the did unto the day:
If I have your for that good good
quickly: your native o.

CORIOLANUS:
As bre get they the y be im, the bere tun me tho the wo gree mere, i 

step  5000 | train 1.313 | val 1.613 | 74.0 min
»  my but I
live, with even in wherein the mat where
I canon colly pited; but your king shill.

LUCIO:
If you love the prisoner, I do not certain with show him,
I consentry weant her and what ange the. y 

step  6000 | train 1.267 | val 1.624 | 90.7 min
»  in a doth king,
Where's he not his face in the heart of my prophecy.
Why, when I dear said all foul marry?

LADY CAPULET:
Camillo Bring him my counsel
Wil he the seem lveden his hals so chard Henown a 
```