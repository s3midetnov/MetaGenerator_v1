Dataset with ~10k entries with added lemmas with expected type of the form
"interpret {3} (at {x4 :: x1 :: x0 :: nil})\n  (var {3} 0 :* var {3} 1 :* (var {3} 2 :* var {3} 0 :* (:inverse {3} (var {3} 0) :* :inverse {3} (var {3} 2) :* :inverse {3} (var {3} 1)))) = interpret {3} (at {x4 :: x1 :: x0 :: nil}) (var {3} 0)"

they come from manually added examples of solutions of trivial equalities in a free group with equation.group meta 
\lemma grequation1{G : Group}(x0 x1 x2 x3 x4 : G): (x4 G.* (G.inverse x4)) = ((x4 G.* (G.inverse x4)) G.* (x4 G.* (((G.inverse x4) G.* x1) G.* (G.inverse x1)))) => equation.group
