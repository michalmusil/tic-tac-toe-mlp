import Generator as gen

dataset = gen.generateBestMoveSet(2000, "X")
gen.saveMovesToJson(dataset, "trainMoves_2000")