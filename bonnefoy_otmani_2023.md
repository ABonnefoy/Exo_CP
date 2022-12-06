
---
title: Modelisation of a Human-Exoskeleton Interaction for Cerebral Palsy
subtitle: 2023 International Symposium on Medical Robotics (ISMR)
author:
- Aurélie BONNEFOY ^1^
- Sabrina OTMANI ^1^,^2^
- <a href="https://gepettoweb.laas.fr/index.php/Members/NicolasMansard">Nicolas MANSARD</a> ^1^
- <a href="https://homepages.laas.fr/ostasse/drupal/">Olivier STASSE</a> ^1^
- <a href="https://ica.cnrs.fr/author/gmichon/">Guilhem MICHON</a> ^2^
- <a href="https://gepettoweb.laas.fr/index.php/Members/BrunoWatier">Bruno WATIER</a> ^1^
org:
- ^1^ Gepetto Team, LAAS-CNRS, France.
- ^2^ MS2M Team, ICA, Toulouse.

code: https://github.com/ABonnefoy/Exo_CP
...

## Abstract

This paper presents a method to model a human-
exoskeleton interaction for patients suffering from cerebral
palsy (CP). More precisely a model of the gait related to
spastic CP is proposed using an optimization program based
on experimental data. The model is done using mechanical
differential equations of motion. A unique feature of this paper
is the Clinical Gait Analysis (CGA) performed on two 9 years
old twin sisters. One has spastic cerebral palsy (C) while the
other is healthy (H) thus without any impairment. This paper
aims at determining the proportion of the walking efforts that
can be supported by the exoskeleton in order to allow a CP
child gait to converge toward a non-pathological one. For this
purpose, minimal torques produced by the human in interaction
with the exoskeleton were studied. The interaction between
the human and the exoskeleton is realized using optimisation
methods such as SLSQP and QuadProg. Ground contacts are
also included in the modelisation. Results show that the human
produces joint torques in the same range of the ones of C.
Exoskeleton succeds in producing additionnal torques to lead
the pathological gait to a non-pathological one. The code for
running the simulations is available on git.


