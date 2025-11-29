import asyncio
from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration, ServerConfiguration

async def main():
    config = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)
    p1 = RandomPlayer(battle_format="gen8randombattle", account_configuration=AccountConfiguration("P1", None), server_configuration=config)
    p2 = RandomPlayer(battle_format="gen8randombattle", account_configuration=AccountConfiguration("P2", None), server_configuration=config)
    
    print("Starting battle...")
    await p1.battle_against(p2, n_battles=1)
    print("Battle finished")

if __name__ == "__main__":
    asyncio.run(main())
