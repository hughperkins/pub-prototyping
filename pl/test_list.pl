use strict;
use Data::Dumper;

print "hello\n";

my @somelist;
push @somelist, 'paris';
push @somelist, 'foo';
push @somelist, 'bar';
print Dumper(@somelist);

printf "%s\n", join(",", @somelist);

