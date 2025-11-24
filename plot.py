import matplotlib.pyplot as plt

x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]  # x轴数据

y1 = [0.9592, 0.9577, 0.9557, 0.9530, 0.9513, 0.9505, 0.9497, 0.9496]
y2 = [0.9578, 0.9566, 0.9553, 0.9522, 0.9505, 0.9491, 0.9486, 0.9482]
y3 = [0.9567, 0.9555, 0.9539, 0.9519, 0.9500, 0.9483, 0.9477, 0.9471]

y4 = [0.9826, 0.9808, 0.9789, 0.9735, 0.9718, 0.9702, 0.9692, 0.9694]
y5 = [0.9812, 0.9798, 0.9783, 0.9727, 0.9708, 0.9687, 0.9685, 0.9680]
y6 = [0.9797, 0.9788, 0.9764, 0.9722, 0.9705, 0.9680, 0.9675, 0.9669]

y7 = [0.9826, 0.9197, 0.9169, 0.9158, 0.9135, 0.9132, 0.9123, 0.9119]
y8 = [0.9812, 0.9182, 0.9166, 0.9147, 0.9125, 0.9113, 0.9102, 0.9098]
y9 = [0.9797, 0.9166, 0.9152, 0.9144, 0.9117, 0.9101, 0.9091, 0.9083]

y10 = [0.9511, 0.9493, 0.9469, 0.9438, 0.9418, 0.9409, 0.9399, 0.9398]
y11 = [0.9495, 0.9480, 0.9465, 0.9428, 0.9408, 0.9391, 0.9385, 0.9380]
y12 = [0.9481, 0.9467, 0.9448, 0.9424, 0.9402, 0.9382, 0.9374, 0.9367]

y13 = [0.0123, 0.0135, 0.0148, 0.0188, 0.0200, 0.0211, 0.0218, 0.0217]
y14 = [0.0133, 0.0142, 0.0153, 0.0194, 0.0207, 0.0222, 0.0223, 0.0227]
y15 = [0.0143, 0.0149, 0.0166, 0.0197, 0.0209, 0.0227, 0.0230, 0.0234]


y16 = [0.9743, 0.9742, 0.9732, 0.9729, 0.9722, 0.9693, 0.9682, 0.9679]
y17 = [0.9738, 0.9727, 0.9716, 0.9699, 0.9688, 0.9677, 0.9674, 0.9666]
y18 = [0.9734, 0.9724, 0.9722, 0.9693, 0.9687, 0.9677, 0.9670, 0.9660]

y19 = [0.9865, 0.9863, 0.9853, 0.9853, 0.9846, 0.9842, 0.9837, 0.9841]
y20 = [0.9856, 0.9851, 0.9851, 0.9846, 0.9840, 0.9835, 0.9835, 0.9831]
y21 = [0.9853, 0.9855, 0.9850, 0.9845, 0.9842, 0.9838, 0.9833, 0.9829]

y22 = 0.9560, 0.9558, 0.9546, 0.9541, 0.9531, 0.9468, 0.9450, 0.9439
y23 = [0.9557, 0.9537, 0.9513, 0.9479, 0.9459, 0.9438, 0.9433, 0.9419]
y24 = [0.9551, 0.9526, 0.9528, 0.9466, 0.9456, 0.9436, 0.9426, 0.9408]

y25 = [0.9710, 0.9708, 0.9697, 0.9694, 0.9686, 0.9652, 0.9640, 0.9635]
y26 = [0.9704, 0.9691, 0.9679, 0.9659, 0.9646, 0.9633, 0.9630, 0.9621]
y27 = [0.9700, 0.9688, 0.9686, 0.9652, 0.9645, 0.9633, 0.9626, 0.9614]

y28 = [0.0106, 0.0107, 0.0115, 0.0116, 0.0121, 0.0123, 0.0127, 0.0124]
y29 = [0.0113, 0.0117, 0.0117, 0.0120, 0.0125, 0.0128, 0.0128, 0.0131]
y30 = [0.0116, 0.0114, 0.0118, 0.0121, 0.0123, 0.0126, 0.0129, 0.0133]

y31 = [0.9566, 0.9559, 0.9554, 0.9541, 0.9535, 0.9516, 0.9498, 0.9472]
y32 = [0.9559, 0.9547, 0.9533, 0.9524, 0.9514, 0.9509, 0.9493, 0.9468]
y33 = [0.9557, 0.9544, 0.9532, 0.9516, 0.9511, 0.9505, 0.9493, 0.9464]

y34 = [0.9363, 0.9350, 0.9338, 0.9326, 0.9316, 0.9278, 0.9256, 0.9222]
y35 = [0.9350, 0.9320, 0.9304, 0.9284, 0.9275, 0.9272, 0.9250, 0.9209]
y36 = [0.9349, 0.9314, 0.9301, 0.9276, 0.9271, 0.9262, 0.9249, 0.9200]

y37 = [0.9125, 0.9114, 0.9109, 0.9075, 0.9064, 0.9033, 0.8993, 0.8935]
y38 = [0.9114, 0.9104, 0.9069, 0.9060, 0.9030, 0.9016, 0.8981, 0.8935]
y39 = [0.9108, 0.9097, 0.9069, 0.9037, 0.9025, 0.9013, 0.8980, 0.8931]

y40 = [0.9243, 0.9230, 0.9222, 0.9199, 0.9188, 0.9154, 0.9122, 0.9076]
y41 = [0.9230, 0.9211, 0.9185, 0.9170, 0.9151, 0.9142, 0.9113, 0.9070]
y42 = [0.9227, 0.9204, 0.9183, 0.9155, 0.9146, 0.9136, 0.9113, 0.9063]

y43 = [0.0253, 0.0258, 0.0263, 0.0267, 0.0271, 0.0286, 0.0295, 0.0307]
y44 = [0.0258, 0.0271, 0.0277, 0.0285, 0.0287, 0.0289, 0.0297, 0.0313]
y45 = [0.0259, 0.0273, 0.0278, 0.0287, 0.0289, 0.0293, 0.0297, 0.0317]


plt.rcParams['font.family'] = 'Times New Roman, SimSun'
fig = plt.figure(figsize=(20, 12))


ax1 = plt.subplot(351)
ax1.plot(x, y1,label=r'$\rho$=0.03')
ax1.plot(x, y2,label=r'$\rho$=0.05')
ax1.plot(x, y3,label=r'$\rho$=0.07')
ax1.set_title('ACC')
ax1.set_xlabel(r'$\rho$')
ax1.set_ylabel('ACC')
ax1.legend()

ax2 = plt.subplot(352)
ax2.plot(x, y4,label=r'$\rho$=0.03')
ax2.plot(x, y5,label=r'$\rho$=0.05')
ax2.plot(x, y6,label=r'$\rho$=0.07')
ax2.set_title('PRC')
ax2.set_xlabel(r'$\rho$')
ax2.set_ylabel('PRC')
ax2.legend()

ax3 = plt.subplot(353)
ax3.plot(x, y7,label=r'$\rho$=0.03')
ax3.plot(x, y8,label=r'$\rho$=0.05')
ax3.plot(x, y9,label=r'$\rho$=0.07')
ax3.set_title('RCL')
ax3.set_xlabel(r'$\rho$')
ax3.set_ylabel('RCL')
ax3.legend()

ax4 = plt.subplot(354)
ax4.plot(x, y10,label=r'$\rho$=0.03')
ax4.plot(x, y11,label=r'$\rho$=0.05')
ax4.plot(x, y12,label=r'$\rho$=0.07')
ax4.set_title('F1')
ax4.set_xlabel(r'$\rho$')
ax4.set_ylabel('F1')
ax4.legend()

ax5 = plt.subplot(355)
ax5.plot(x, y13,label=r'$\rho$=0.03')
ax5.plot(x, y14,label=r'$\rho$=0.05')
ax5.plot(x, y15,label=r'$\rho$=0.07')
ax5.set_title('FPR')
ax5.set_xlabel(r'$\rho$')
ax5.set_ylabel('FPR')
ax5.legend()

ax6 = plt.subplot(356)
ax6.plot(x, y16,label=r'$\rho$=0.03')
ax6.plot(x, y17,label=r'$\rho$=0.05')
ax6.plot(x, y18,label=r'$\rho$=0.07')
ax6.set_title('ACC')
ax6.set_xlabel(r'$\rho$')
ax6.set_ylabel('ACC')
ax6.legend()

ax7 = plt.subplot(357)
ax7.plot(x, y19,label=r'$\rho$=0.03')
ax7.plot(x, y20,label=r'$\rho$=0.05')
ax7.plot(x, y21,label=r'$\rho$=0.07')
ax7.set_title('PRC')
ax7.set_xlabel(r'$\rho$')
ax7.set_ylabel('PRC')
ax7.legend()

ax8 = plt.subplot(358)
ax8.plot(x, y22,label=r'$\rho$=0.03')
ax8.plot(x, y23,label=r'$\rho$=0.05')
ax8.plot(x, y24,label=r'$\rho$=0.07')
ax8.set_title('RCL')
ax8.set_xlabel(r'$\rho$')
ax8.set_ylabel('RCL')
ax8.legend()

ax9 = plt.subplot(359)
ax9.plot(x, y25,label=r'$\rho$=0.03')
ax9.plot(x, y26,label=r'$\rho$=0.05')
ax9.plot(x, y27,label=r'$\rho$=0.07')
ax9.set_title('F1')
ax9.set_xlabel(r'$\rho$')
ax9.set_ylabel('F1')
ax9.legend()

ax10 = plt.subplot(3,5,10)
ax10.plot(x, y28,label=r'$\rho$=0.03')
ax10.plot(x, y29,label=r'$\rho$=0.05')
ax10.plot(x, y30,label=r'$\rho$=0.07')
ax10.set_title('FPR')
ax10.set_xlabel(r'$\rho$')
ax10.set_ylabel('FPR')
ax10.legend()

# 绘制折线图
ax11 = plt.subplot(3,5,11)
ax11.plot(x, y31,label=r'$\rho$=0.03')
ax11.plot(x, y32,label=r'$\rho$=0.05')
ax11.plot(x, y33,label=r'$\rho$=0.07')
ax11.set_title('ACC')
ax11.set_xlabel(r'$\rho$')
ax11.set_ylabel('ACC')
ax11.legend()

ax12 = plt.subplot(3,5,12)
ax12.plot(x, y34,label=r'$\rho$=0.03')
ax12.plot(x, y35,label=r'$\rho$=0.05')
ax12.plot(x, y36,label=r'$\rho$=0.07')
ax12.set_title('PRC')
ax12.set_xlabel(r'$\rho$')
ax12.set_ylabel('PRC')
ax12.legend()

ax13 = plt.subplot(3,5,13)
ax13.plot(x, y37,label=r'$\rho$=0.03')
ax13.plot(x, y38,label=r'$\rho$=0.05')
ax13.plot(x, y39,label=r'$\rho$=0.07')
ax13.set_title('RCL')
ax13.set_xlabel(r'$\rho$')
ax13.set_ylabel('RCL')
ax13.legend()

ax14 = plt.subplot(3,5,14)
ax14.plot(x, y40,label=r'$\rho$=0.03')
ax14.plot(x, y41,label=r'$\rho$=0.05')
ax14.plot(x, y42,label=r'$\rho$=0.07')
ax14.set_title('F1')
ax14.set_xlabel(r'$\rho$')
ax14.set_ylabel('F1')
ax14.legend()

ax15 = plt.subplot(3,5,15)
ax15.plot(x, y43,label=r'$\rho$=0.03')
ax15.plot(x, y44,label=r'$\rho$=0.05')
ax15.plot(x, y45,label=r'$\rho$=0.07')
ax15.set_title('FPR')
ax15.set_xlabel(r'$\rho$')
ax15.set_ylabel('FPR')
ax15.legend()

fig.text(0.05, 0.85, 'NSL-KDD', va='center', rotation='vertical')
fig.text(0.05, 0.5, 'UNSW-NB15', va='center', rotation='vertical')
fig.text(0.05, 0.15, 'CIC-IDS-2017', va='center', rotation='vertical')

plt.tight_layout()
plt.subplots_adjust(left=0.1)

plt.show()
