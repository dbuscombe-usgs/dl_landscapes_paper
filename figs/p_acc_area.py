import matplotlib.pyplot as plt

s_A = [5873,	4886,	290,	5071,	469,	9268,	1307]
s_P = [0.87,	0.87,	0.44,	0.95,	0.63,	0.98,	0.86]
s_R = [0.95,	0.93,	0.51,	0.92,	0.41,	0.92,	0.81]
s_F = [0.9,	0.9,	0.46,	0.93,	0.48,	0.95,	0.83]


o_A = [1454,	860,	16042,	7354,	3655]
o_P = [0.72	,0.32,	0.95,	0.9,	0.78]
o_R = [0.72	,0.31,	0.95,	0.93,	0.59]
o_F = [0.74	,0.3,	0.95,	0.89,	0.64]


g_A = [9728,	2138,	1576,	6906]
g_P = [0.8,	0.94,	0.92,	0.76]
g_R = [0.97,	0.92,	0.31,	0.79]
g_F = [0.87,	0.93,	0.46,	0.8]

plt.subplot(221)
plt.semilogx(s_A, s_F, 'ko', label='Seabright')
plt.plot(o_A, o_F, 'rs', label='Ontario')
plt.plot(g_A, g_F, 'bh', label='Grand Canyon')
plt.ylabel('Mean F1 score', fontsize=6)
plt.xlabel('Mean Area (px$^2$)', fontsize=6)
plt.legend(fontsize=6, loc=4)
#plt.show()
plt.setp(plt.xticks()[1], fontsize=6)  
plt.setp(plt.yticks()[1], fontsize=6)
plt.savefig('F_vs_area.png', dpi=300, bbox_inches='tight')
plt.close()  
