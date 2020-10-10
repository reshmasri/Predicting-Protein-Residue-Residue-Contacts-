# Predicting-Protein-Residue-Residue-Contacts

Residue-residue contact prediction is predicting which amino acid
residues in the structure of a protein are “in contact”. Typically, residues are
defined to be in contact when the distance between their α-carbon atoms is
smaller than 8Å. Usually, contact prediction methods are structured as either
sequence-based or template-based. Sequence-based contact prediction
normally uses machine learning methods. This project comprises of residueresidue contact prediction using machine learning models - random forest
and logistic regression classifier. The primary goal of contact prediction here
is typically to produce predictions that correctly label a pair of amino acids
in a protein as “in contact” or “not in contact”. The ability to predict which
pairs of amino acid residues in a protein are in contact with each other offers
many advantages for various areas of research that focus on proteins. There
have been several approaches to achieve this, but we research on defining
one such framework that will predict the contact using one hot encoding
obtained from the window of amino acids and contact maps.
