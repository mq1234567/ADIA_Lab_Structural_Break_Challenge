## Breakpoint localisation
- problem: windowed-statistics can contain data points before the break as well as after the break, contaminating the stats
	- so need a better mechanism for choosing such windows
	- a localiser: to propose where to cut the window, a cut point $t_k$
- devision of labour, the overall detector is the combination of
	1. a localiser - propose where to cut (not trustworthy on its own)
	2. statistics on $[t_k, t]$, as features
	3. decides whether the difference is signal/noice
## DL/PFM experiments


## Smoothed Tau (auxiliary targets)