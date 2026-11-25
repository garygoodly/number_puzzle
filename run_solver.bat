set BROWSER=msedge
set SIZE=6
set NAME=Browniebro
set ASTAR_WEIGHT=1.0
set MOVE_DELAY_MS=18

python solve_web.py --browser %BROWSER% --size %SIZE% --name %NAME% --astar-weight %ASTAR_WEIGHT% --move-delay-ms %MOVE_DELAY_MS%
