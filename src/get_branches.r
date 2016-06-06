# gets the branch lengths for HC1
# install.packages("rphast") # to install package
require("rphast") # to load package
# read in tree
tree <- paste(scan("../data/tree.nh", what="character"), sep="", collapse="")
# prune tree
prunedTree <- prune.tree(tree, c("hg19", "panTro2", "gorGor1", "rheMac2"), all.but=TRUE)
prunedTree
