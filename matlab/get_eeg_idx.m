function eeg_idx = get_eeg_idx(subject)

    if strcmp(subject,'A1974')
            % 23.3 ms
            eeg_idx = 149;
    elseif strcmp(subject,'A1999')
            % 22.5 ms
            eeg_idx = 148;
    elseif strcmp(subject,'A0206')
        %25 ms
        eeg_idx = 151;      

    end
end

