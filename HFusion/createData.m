function collected_reports = createData(events, m_reported_duration, overlap_rate)
% collect multiple, possibly overlapping reports about  number of events within
% reported intervals
	collected_reports = [];

	% report begin from the first.
	from = 1;
	max_time = length(events);

	% generate report by shift the from until from larger then the max.
	while from <= max_time   
	  tmp = from + m_reported_duration-1;
	  
	  if tmp <= max_time 
		to = tmp;
	  else 
		to = max_time;
	  end
	  
	  collected_reports = [collected_reports; gen_report(events, from, to)]; %%% total number of events

%  	  if to == max_time
%  		break;
% 	  end
	  
	  from = from + overlap_rate;

	end  
end
