function source_activation_mri_alt(mri,scale,dat,pos,thresh,time,tit)
% Plot MRI slices and the source activation
% A variaton of source_activation_mri as it plots only sagittal and axial
% mri: fieldtrip mri structure
% scale: scale value
% dat: number of sources x 3 values
% pos: source space points;
% thresh: threshold value of the source activity
% time: the time interval of the represented activity
% tit: the title of the inverse method

% f = figure('WindowScrollWheelFcn',@figScroll,'KeyPressFcn',@figButton,...
%     'WindowButtonUpFcn',@ImageClickUp,'Name','Scroll MRI');
f = figure('WindowScrollWheelFcn',@figScroll,'WindowButtonUpFcn',@ImageClickUp,'Name','Source Activity on MRI',...
    'pos',[200 200 1200 700]);
%'pos',[100 100 450 400]

% make source space positive
for i = 1:size(pos,2)
    if any(pos(:,i)<1); pos(:,i) = 256+pos(:,i); end
end

if length(pos) == round(length(dat)/3)
    dat = reshape(dat,3,size(dat,1)/3)';
end

dat = sqrt(sum(dat.^2,2));

dat(dat < max(dat)*thresh) = NaN;

%dat(dat < max(dat)*thresh) = 0;

[~,idm] = max(dat);

pos = round(pos);
xx = linspace(1,mri.dim(1),mri.dim(1));
yy = linspace(1,mri.dim(2),mri.dim(2));
zz = linspace(1,mri.dim(3),mri.dim(3));

sagittal_pos = [.05 .45 .45 .45];
%coronal_pos  = [.36 .55 .3 .3];
axial_pos    = [.52 .45 .45 .45];

count = pos(idm,:);

%dat = (dat - min(dat)) / ( max(dat) - min(dat));

img = nan*ones(length(xx),length(yy),length(zz));
z1 = unique(pos(:,3));
% for i = 1:length(z1)
%     idxx = find((pos(:,3) == z1(i))==1);
%     for j = 1:length(idxx)
%         img(pos(idxx(j),1),pos(idxx(j),2),pos(idxx(j),3))=dat(idxx(j));
%     end
% end

plot_data(mri,scale,dat,pos,count,time)

    function figScroll(src,evnt)
        
        dm = mri.dim(1);
        
        if isprop(evnt,'VerticalScrollCount')
            scl = evnt.VerticalScrollCount;
        else
            scl = NaN;
        end
        if ~isnan(scl)
            if scl < 0
                if count < dm
                    count=count+2;
                else
                    count = dm;
                end
            elseif scl > 0
                if count > 1
                    count=count-2;
                else
                    count=1;
                end
            end
            plot_data(mri,scale,dat,pos,count,time)
        end
    end
    function figButton(src,evnt)
        
        dm = mri.dim(1);
        
        %dm2 = mri.dim(2);
        
        %dm3 = mri.dim(3);
        
        if isprop(evnt,'Key')
            bt = double(get(gcf,'CurrentCharacter'));
        else
            bt = 1;
        end
        %disp(event.Key)
        %if isempty(bt); bt = 1; end
        
        if bt == 30
            if count < dm
                count=count+2;
            else
                count = dm;
            end
        elseif bt == 31
            if count > 1
                count=count-2;
            else
                count=1;
            end
        end
        plot_data(mri,scale,dat,pos,count,time)
    end

    function ImageClickUp ( src , evnt )
        axesHandle  = get(src,'CurrentAxes');
        coordinates = get(axesHandle,'CurrentPoint');
        coordinates = coordinates(1,1:2);
        rcoords     = round(coordinates);
        
        %         if mod(rcoords(1),2) ~= 0
        %             rcoords(1)=rcoords(1)+1;
        %         end
        %         if mod(rcoords(2),2) ~= 0
        %             rcoords(2)=rcoords(2)+1;
        %         end
        
        ax_pos = axesHandle.Position;
        if round(pdist2(ax_pos,sagittal_pos),1) == 0
            count2 = [count(1) rcoords(1) mri.dim(3) - rcoords(2)];
        elseif round(pdist2(ax_pos,axial_pos),1) == 0
            count2 = [rcoords(1) mri.dim(2) - rcoords(2) count(3)];
        end
        if count2(1) > 0 && count2(1) < mri.dim(1) && ...
                count2(2) > 0 && count2(2) < mri.dim(2) && ...
                count2(3) > 0 && count2(3) < mri.dim(3)
            count = count2;
            plot_data(mri,scale,dat,pos,count,time)
        end
    end
    function plot_data(mri,scale,dat,pos,count,time)
        clf; % clc;
        
        %sagital view
        fss = 16;
        
        idx=find(pos(:,1)==count(1));
        
        ax1 = axes('Parent',f,'Units','normalized','Position',sagittal_pos);
        ax2 = axes('Parent',f,'Units','normalized','Position',sagittal_pos);
        ax3 = axes('Parent',f,'Units','normalized','Position',sagittal_pos);
        
        if mod(count(1),2)==1
            if count(1) < mri.dim(1)
                count(1)=count(1)+1;
            else
                count(1)=count(1)-1;
            end
        end
        
        %contrast-enhancement img
        adj_img = cont_enh(squeeze(mri.anatomy(count(1),:,:)));%cont_enh
        
        %show img
        imagesc(ax1,yy,zz,rot90(adj_img));
        
        if length(find(dat~=0)) == 1
            scatter(ax2,pos(idx,2),pos(idx,3),[],dat(idx,:),'s','filled');
        else
            scatter(ax2,pos(idx,2),pos(idx,3),scale*dat(idx,:),dat(idx,:),'s','filled');
        end
        im2 = imagesc(ax3,yy,zz,nan(mri.dim(2),mri.dim(3)));  set(im2, 'AlphaData', 0)
        text(20,mri.dim(2)-20,'P/I','color','w','fontweight','bold','fontsize',fss)
        text(mri.dim(2)-20,mri.dim(3)-20,'A','color','w','fontweight','bold','fontsize',fss)
        text(20,20,'S','color','w','fontweight','bold','fontsize',fss)
        %text(mri.dim(2)-50,20,[num2str(floor(1000*time)) ' ms'],'color','w','fontweight','bold','fontsize',fss)
        title(['Sagittal ' num2str(count(1)) '/' num2str(mri.dim(1))],'fontsize',fss)
        
        axis([1 mri.dim(2) 1 mri.dim(3)])
        axis off
        
        % Link them together
        linkaxes([ax1,ax2,ax3])
        % Hide the top axes
        ax1.Visible = 'off';
        ax2.Visible = 'off';
        ax3.XTick = [];
        ax3.YTick = [];
        % Give each one its own colormap
        colormap(ax1,'gray')
        colormap(ax2,'autumn')
        colormap(ax3,'autumn')
        
       
        
        % Link them together
        linkaxes([ax1,ax2,ax3])
        % Hide the top axes
        ax1.Visible = 'off';
        ax2.Visible = 'off';
        ax3.XTick = [];
        ax3.YTick = [];
        % Give each one its own colormap
        colormap(ax1,'gray')
        colormap(ax2,'autumn')
        colormap(ax3,'autumn')
        
        % axial view
        idx=find(pos(:,3)==count(3));
        
        ax1 = axes('Parent',f,'Units','normalized','Position',axial_pos);
        ax2 = axes('Parent',f,'Units','normalized','Position',axial_pos);
        ax3 = axes('Parent',f,'Units','normalized','Position',axial_pos);
        
        
        if mod(count(3),2)==1
            if count(3) < mri.dim(3)
                count(3)=count(3)+1;
            else
                count(3)=count(3)-1;
            end
        end
        
        %contrast-enhancement img
        adj_img = cont_enh(squeeze(mri.anatomy(:,:,count(3))));%cont_enh
        
        %show img
        imagesc(ax1,xx,yy,rot90(adj_img));
        if length(find(dat~=0)) == 1
            scatter(ax2,pos(idx,1),pos(idx,2),[],dat(idx,:),'s','filled');
        else
            scatter(ax2,pos(idx,1),pos(idx,2),scale*dat(idx,:),dat(idx,:),'s','filled');
        end
        im2 = imagesc(ax3,xx,yy,nan(mri.dim(1),mri.dim(2)));  set(im2, 'AlphaData', 0)
        text(20,mri.dim(3)-20,'L/P','color','w','fontweight','bold','fontsize',fss)
        text(mri.dim(1)-20,mri.dim(3)-20,'R','color','w','fontweight','bold','fontsize',fss)
        text(20,20,'A','color','w','fontweight','bold','fontsize',fss)
        title(['Axial ' num2str(count(3)) '/' num2str(mri.dim(3))],'fontsize',fss)
        
        axis([1 mri.dim(1) 1 mri.dim(2)])
        axis off
        
        % Hide the top axes
        linkaxes([ax1,ax2,ax3])
        % Hide the top axes
        ax1.Visible = 'off';
        ax2.Visible = 'off';
        ax3.XTick = [];
        ax3.YTick = [];
        % Give each one its own colormap
        colormap(ax1,'gray')
        colormap(ax2,'autumn')
        colormap(ax3,'autumn')
        %hcb=colorbar(ax3,'Position',[.52 .05 .05 .4]);
        %ylabel(hcb,tit,'fontsize',fss)
        
        %         ax1 = axes('Parent',f,'Units','normalized','Position',[.6 .1 .4 .4]);
        %         para = []; para.title = ['sLORETA']; para.tt=eye(3); para.axis_handle=ax1;
        %         thr=0.5;
        %         alpha=.5;
        %         plot_slice=0;
        %         plot_dipole=0;
        %         plot_inv_on_surf(model,para,u_sLORETA,thr,pos,mri,alpha,plot_slice,plot_dipole)
        
    end
    function adj_img = cont_enh(img)
        img     = (img - min(min(img)))/(max(max(img))-min(min(img))); % image normalization 
        adj_img = adapthisteq(img);
    end   
end