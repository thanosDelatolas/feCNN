function [] = saver( path, question, num, offset )
%saves figures to the given path!    
    for i = 1:num
        figure(i);
        %set(gcf,'units','inches','position',[0,0,6.5,4.2+1*(i==1)])
        saveas(gcf,sprintf('%sfig_%d_%d',path,question,i+offset),'png');
    end
end