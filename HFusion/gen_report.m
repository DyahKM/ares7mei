function report = gen_report(events, from, to)
% generates reports counting number of events within the [from,to]
% interval
report = [from, to, sum(events(from:to))];

end

