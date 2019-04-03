import json
import neat 

class JsonThotPersist:

    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                '../config_trader')
    def __init__(self):
        return

    def saveAsJson(self):
        jsonObj = json.dumps(str(self.config.config_dict))
        
        with open('deepthotconfig.json', 'w') as fp:
            json.dump(json.loads(jsonObj), fp)

    def loadFromJson(self):
        with open('deepthotconfig.json') as jfile:
            return json.load(jfile)

jsonPersister = JsonThotPersist()
print(jsonPersister.loadFromJson())