# DL partitions
my @partitions = grep /^dl/, `sinfo`;

# Idle nodes
my @idle_parts = grep /idle/, @partitions;

# Set to dl_shared for now
# Change here for desired partition
print "dl_shared";

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
