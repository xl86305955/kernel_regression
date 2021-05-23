All:
	python run_experiment.py mnist nn DEEPFOOL

halfmoon_fgsm:
	python run_experiment.py halfmoon nn FGSM
	python plot.py halfmoon FGSM

halfmoon_deepfool:
	python run_experiment.py halfmoon nn DEEPFOOL
	python plot.py halfmoon DEEPFOOL

halfmoon_cw:
	python run_experiment.py halfmoon nn CW
	python plot.py halfmoon CW

halfmoon_pgd:
	python run_experiment.py halfmoon nn PGD
	python plot.py halfmoon PGD

abalone_fgsm:
	python run_experiment.py abalone nn FGSM
	python plot.py abalone FGSM

abalone_deepfool:
	python run_experiment.py abalone nn DEEPFOOL
	python plot.py abalone DEEPFOOL

abalone_pgd:
	python run_experiment.py abalone nn PGD
	python plot.py abalone PGD

abalone_cw:
	python run_experiment.py abalone nn CW
	python plot.py abalone CW

mnist_fgsm:
	python run_experiment.py mnist nn FGSM
	python plot.py mnist FGSM

mnist_deepfool:
	python run_experiment.py mnist nn DEEPFOOL
	python plot.py mnist DEEPFOOL

mnist_cw:
	python run_experiment.py mnist nn CW
	python plot.py mnist CW

mnist_pgd:
	python run_experiment.py mnist nn PGD
	python plot.py mnist PGD

mnist:
	python run_experiment.py mnist nn FGSM
	python run_experiment.py mnist nn PGD
	python run_experiment.py mnist nn DEEPFOOL
	python run_experiment.py mnist nn CW

ab:
	python run_experiment.py abalone nn FGSM 100
	python run_experiment.py abalone nn PGD 100
	python run_experiment.py abalone nn DEEPFOOL 100
	python run_experiment.py abalone nn FGSM 250
	python run_experiment.py abalone nn PGD 250
	python run_experiment.py abalone nn DEEPFOOL 250
	python run_experiment.py abalone nn CW 250
	python run_experiment.py abalone nn CW 100

halfmoon:
	python run_experiment.py halfmoon nn FGSM 100
	python run_experiment.py halfmoon nn PGD 100
	python run_experiment.py halfmoon nn DEEPFOOL 100
	python run_experiment.py halfmoon nn FGSM 250
	python run_experiment.py halfmoon nn PGD 250
	python run_experiment.py halfmoon nn DEEPFOOL 250
	python run_experiment.py halfmoon nn CW 250
	python run_experiment.py halfmoon nn CW 100

#halfmoon:
#	python run_experiment.py halfmoon nn FGSM
#	python run_experiment.py halfmoon nn PGD
#	python run_experiment.py halfmoon nn DEEPFOOL
#	python run_experiment.py halfmoon nn CW

abalone_cw:
	python run_experiment.py abalone nn CW 20
	python run_experiment.py abalone nn CW 50
	python run_experiment.py abalone nn CW 500

