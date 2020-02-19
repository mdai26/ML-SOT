# Machine Learning Prediction of Field-Free Spin-Orbit-Torque Switching
## data
1069 micromagnetic simulation are performed with different disk diameter, interfacial DMI strength D and the charge current. For regression, the target is the z component of final average magentization. For classification, the target is 1 (\<m<sub>z</sub>\> <-0.1), 2 (-0.1 < \<m<sub>z</sub>\> < 0.1) or 3 (\<m<sub>z</sub>\> > 0.1)
## code
The datasets are divided into 989 training groups and testing groups. Differnt regression methods and classification methods are perfomed using Scikit-learn. The accuracy of regression is represented by R<sup>2</sup> score and the dicision tree regression model delivers the highest accuracy.
