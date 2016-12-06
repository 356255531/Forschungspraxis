CartPole: clean
	python CartPole.py
	rm -f Toolbox/*.pyc

MountainCar: clean
	python MountainCar.py
	rm -f Toolbox/*.pyc

clean:
	rm -f Toolbox/*.pyc
