import duplication_analysis as dp

y=[1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,1]
y_real=[1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,0,1]

dp.plot_statistics(y,y_real,"test")

y=[1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,1]
y_real=[1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,0,1]

dp.plot_statistics(y,y_real,"test2")