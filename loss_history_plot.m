cost_list = [str2num(fileread('cost_history.txt'))];

figure();
x = 1:length(cost_list);
plot(x, cost_list);
grid on
xlabel('Epoch');
ylabel('Loss');