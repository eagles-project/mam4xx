# a100 partitions
my @partitions = grep /^a100/, `sinfo`;

# Idle nodes
my @idle_parts = grep /idle/, @partitions;

# Set to a100_shared for now
# Change here for desired partition
print "a100_shared,a100_80_shared,a100,a100_80";

# Logic for selecting dl(v)(t) partitions with idle nodes
#if ( scalar @idle_parts eq 0 ) {
#
#  # No idle dl parititons... Just choose the shared one.
#  print "dl_shared";
#}
#else {
#  my @parts = split / /, $idle_parts[0];
#  print $parts[0];
#}
