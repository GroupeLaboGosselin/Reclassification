# Reclassification

The function in this repository helps improve the signal-to-noise ratio (SNR) in psychological experiments that use accuracy as a selection variable for another dependent variable (e.g., when comparing brain activity for correct vs incorrect trials, or when computing classification images). Specifically, the function reclassifies correct responses that are likely to be correct only by chance (false correct responses) as incorrect responses using a trial-by-trial reclassification evidence, such as response time. The procedure is conservative in the sense that if the reclassification evidence is uninformative, no correct response will be reclassified as incorrect. The more difficult the task and the fewer the response alternatives, the more to be gained from the procedure. Reclassification improved SNR by 13-20% on existing behavioral and ERP data. 

![Figure](/reclassif_fig1.png)

Please cite our paper if you use this function:

Gosselin, F., Daigneault, V., Larouche, J.-M, & Caplette, L. (submitted). Reclassifying guesses to increase signal-to-noise ratio in psychological experiments. <psyarxiv.com/3s9du>
