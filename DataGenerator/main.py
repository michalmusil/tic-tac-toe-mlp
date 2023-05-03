# Author: Michal Musil
import Generator as gen

dataset = gen.generateBestMoveSet(4519, "X")
gen.saveMovesToJson(dataset, "trainMoves2")