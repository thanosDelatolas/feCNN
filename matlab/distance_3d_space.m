function dist = distance_3d_space(point_1,point_2)
% Returns the distance in the 3d space between point_1 and point_2.

 dist = sqrt((point_1(1)-point_2(1)).^2 + (point_1(2)-point_2(2)).^2 + ...
     (point_1(3)-point_2(3)).^2 );

end

