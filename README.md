# cryptochronolonic
crypto bot that runs on the poloniex exchange - implements machine learning
currently adding a hyperneat genetic encoded nn and fitness task optimized for weekly gains

THE SETUP:

First create a folder and cd into via the command line, then you will want to make a python virtual environment. Once that is done you can "git clone https://github.com/nickwilliamsnewby/cryptochronolonic.git " cd in and do 
git checkout latest. Then run pip install -r requirements.txt and then cd .. back out of the folder.
then you also need to clone neat-python (https://github.com/nickwilliamsnewby/neat-python.git) cd into it and do a "git checkout python3-rnn-fix" before running "python setup.py build", followed by "python setup.py install".
Now cd .. back into the folder you created at the started you need to clone https://github.com/nickwilliamsnewby/pureples.git cd into it run git checkout latests then again we run 
"python setup.py build", followed by "python setup.py install". 
Now you should be setup and ready to RIDE, but how does one do such proverbial running you may ask, will sweet child of mine i have an answer. cd into the cryptochronolonic directory and then also pull it up in your fav text editor, open the file trading_purples.py, At the top there are some evolution related configs in a dictionary called config which you can edit if you understand what the implications are otherwise just leave it be. So if you want to just run some training sessions you can modify the size, elitetism and some other more intuitive settings for the evo process in the config_trader and then set the length of the time the net trades 
to evaluate fitness in trading_purples.py on line 60 (very bottom of the init constructor), this is the amount of two hour bars it will be looped through and evaluated. then you can just set the number of generations down on line 147 (this may change so if its not there its set in the second line of the "main" function for the file) and run the file, it should start training and if not, write up a bit about your setup and the error you recieve and open an issue and ill do my best to help you out, once training is done you can create some polo api keys 
and plug them into brain_trader.py and run that to start #aialgotrading 
Cheers
