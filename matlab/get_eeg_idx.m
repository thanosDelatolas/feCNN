function eeg_idx = get_eeg_idx(subject,ms)

    if strcmp(subject,'A1974')
        if strcmp(ms,'22_5')
            % 22.5 ms
            eeg_idx = 148;
        elseif strcmp(ms,'23_3')
            % 23.3 ms
            eeg_idx = 149;
        elseif strcmp(ms,'24_2')
            % 24.2 ms
            eeg_idx = 150;
        elseif strcmp(ms,'25')
            % 25 ms
            eeg_idx = 151;
        end
    elseif strcmp(subject,'A1999')

        if strcmp(ms,'17_5')
            eeg_idx = 142;
        elseif strcmp(ms,'21_7')
            eeg_idx = 147;
        elseif strcmp(ms,'22_5')
            eeg_idx = 148;
        elseif strcmp(ms,'23')
            eeg_idx = 157;
        elseif strcmp(ms,'20_8')
            eeg_idx = 146;
        end

    elseif strcmp(subject,'A0206')    
        if strcmp(ms,'20')
                eeg_idx = 145;
        elseif strcmp(ms,'24_2')
                eeg_idx = 150;
        elseif strcmp(ms,'25')
            eeg_idx = 151;
        elseif  strcmp(ms,'25_8')
            eeg_idx = 152;
        end

    end
end

