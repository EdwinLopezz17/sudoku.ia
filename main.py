import numpy as np
from DQNAgent import DQNAgent
from SudokuEnv import SudokuEnv



# Crear agente y cargar modelo
agent = DQNAgent()

# Ruta del modelo entrenado
model_path = "models/model_sudoku_ia_ep1900.keras"
agent.load(model_path)
agent.epsilon = 0.0  # Desactiva exploración (solo acciones óptimas)

# Puzzle a resolver
puzzle = np.array([
    [0, 4, 0, 9, 0, 0, 3, 0, 8],
    [2, 0, 8, 0, 0, 4, 0, 6, 1],
    [6, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 4, 0, 0, 0, 7],
    [7, 6, 0, 0, 3, 1, 9, 8, 0],
    [0, 0, 9, 8, 2, 0, 0, 1, 0],
    [0, 2, 0, 0, 0, 3, 0, 0, 0],
    [9, 3, 6, 2, 1, 0, 7, 4, 0],
    [0, 0, 7, 0, 9, 6, 8, 3, 0]
])

# Crear entorno y cargar el puzzle
env = SudokuEnv()
state = env.reset(puzzle)

# Ejecutar el modelo para resolver el Sudoku
MAX_STEPS = 81
done = False
steps = 0

print("📋 Sudoku Inicial:")
env.render()

while not done and steps < MAX_STEPS:
    mask = env.get_action_mask()
    constraints = env.get_number_constraints()

    action_index = agent.act(state, mask, constraints)
    row, col, num = agent.decode_action(action_index)

    state, reward, done, info = env.step((row, col, num))
    steps += 1

    if steps % 10 == 0 or done:
        print(f"\n🔁 Estado después de {steps} pasos:")
        env.render()

print("\n✅ Sudoku Final:")
env.render()

# Verifica si el Sudoku está correctamente resuelto
if env.check_done():
    print("🎉 ¡Sudoku resuelto exitosamente!")
else:
    print("⚠️ El modelo no pudo resolver el Sudoku completamente.")
