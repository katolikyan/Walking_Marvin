# Walking_Marvin

Open AI gym project.

### Instalation

```bash
$> pip3 install swig box2d-py ansicolors gym
```

```bash
$> git clone https://github.com/katolikyan/Walking_Marvin.git
$> cd Walking_Marvin/gym-0.12.0
$> pip3 install -e .
```
> Install all other dependencies if needed.

### Usage

```bash
$> python3 Marvin.py
```

 #### Available commands:

 * `–-walk` or `-w`
   > Display only walking process.

 * `–-help`  or  `-h`  
   > Display available commands.

 * `–-load`  or  `-l`  
   > File Load weights for Marvin agent from a file. Skip training process if this option is specified.

 * `–-logs`  or  `-L`  
   > Display training logs.

 * `–-save`  or  `-s`  
   > File Save weights to a file after running the program.

 * `–-walking-logs`  or  `-wl`  
   > Display logs while walking.


 :

 `python3 Marvin.py -L -l actions.npy -s`
   > Will load weights from file, continue training based on that weights and save new best performing weights

---

### Examples
```bash
$> python3 Marvin -w
...
$> python3 Marvin -l actions.npy -wl -w
...
```

![](https://github.com/katolikyan/Walking_Marvin/blob/master/.media/Marvin_run.gif)

```bash
$> python3 Marvin -L -s
...
```

![](https://github.com/katolikyan/Walking_Marvin/blob/master/.media/Marvin_training_logs.png)

---
### Future plans

* Create more afficiant neeural networks.
