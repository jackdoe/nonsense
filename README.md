```
def _unitary(self, h_real):
    H = 0.5*(A + A^T)                # make a symmetric real matrix
    U = expm(−i·H·dt)               # a learnable unitary via matrix exponential
    return h_c + (h_c @ U)
h_c = self._unitary(self.ln2(h_c.real).to(torch.cfloat))
```

experimenting with unitary feature mixer `U = e^{−iHΔt}` 



```
step     0 | train 4.347 | val 4.344 | 0.0 min
»  bTj?ox 3P$rVNHf!b,R,hrx,rgsMTjN-fmvQkCUX!&mKwzgHn?K!xxRVeCOvx;Vprs:
vPlV pnvzq$KyW;xDO&ilfP-y' fgBwmJuNxXyglCYtf:lN--DSY z:OUa
WHevocC?CkKm,KKw.z.ystl.LW
azKzPIAsTupJevNjL-?WFm3$gF,K-lqFLzduswvvVXhcaC 

step  1000 | train 1.687 | val 1.882 | 1.9 min
»  crarishsh poastel met of before:
Speak leave 'Out in
Eve ben it the rother go sin.

KING EDWARD:
Henon pirt nd this thaver m-harn lath's he m thonon:
Thim drothireran igh hends we dad
Anderre the s ar 

step  2000 | train 1.534 | val 1.737 | 3.5 min
»  hands consulined to kind,
Shome vengery repled afer which the traitor,
Which illd he ar his wars not ot the his it agamous t
mand ifus monce ablar wat mor;
Thingor man bus thefouge ars.
Ton t I then t 

step  3000 | train 1.475 | val 1.669 | 5.1 min
»  this serve, he armity come more:
Here you office with himself-ancts! I will palace it;
For are hence, and lawstert spir, by ulleanck bone t fin
Thin Le t horour hasit he
Anf-he overe was t I wous t
Ti 

step  4000 | train 1.441 | val 1.628 | 6.7 min
»  farewel
As none, my royal stands to the will turn of this bold.

Second MONIUS:
Friar of then, with them
Welll it your yourses
Thialfincy s Clare Parence s titie;
I as 'Tis s teeeee

Tun ald
See Musss 

step  5000 | train 1.403 | val 1.625 | 8.2 min
»  time, the request.

ROMEO:
My lord? fix Is all enry thy vow most aged before it,
Ther yor the to is stant, teant tuld dowh!
What t the coung, ant tist.
Mot y urecheastst:
Sesty aist orest, tore-lit ad 

step  6000 | train 1.387 | val 1.612 | 9.8 min
»  may the clates: the foret, if thou art my
pity could to with the one cold that sin
the wither off the piesit ledger thee.

Which thest to roudds we to I incher, on cheanquit phaty wis tranched.
Therdo 

step  7000 | train 1.380 | val 1.596 | 11.4 min
»  III:
Alas, if thy kindred well.

ESCALUS:
King Hereter.
Takes and them you take it brought ather's partt the never kinged low'd twark goch.

Ond this killlow wit how, as wratchoooord I's:
And was morn 

step  8000 | train 1.345 | val 1.612 | 12.9 min
»  breakch makes my one
To order my brought my father love.

RATCLIF:
Then was not by that mean delith it,
An't for I held to no the deeet tremaslesss
Torane.

Afe. Whore Whorfe llle'd forest oree tchrin 

step  9000 | train 1.342 | val 1.561 | 14.5 min
»  some solely like flint.

CORIOLANUS:
This is allial the thing eyes on man of you.


RICHARD:
Welll, go in alafe, wakerthingess manine He there thare ath, we anve thicee thins ars owath st peent.

step 10000 | train 1.336 | val 1.538 | 16.1 min
»  have himiding.

DUKE OF YORK:
From Northumberland, from their days to flight one;
This if not defy this lood:
And beiligng the bought too thes,
Thavous nder themine hll kus ents in aflll ore tt on un: 
```