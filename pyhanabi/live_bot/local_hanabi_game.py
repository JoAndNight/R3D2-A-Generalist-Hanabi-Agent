#!/usr/bin/env python3
"""
本地终端Hanabi对战程序
直接与训练好的AI模型对战，无需服务器
"""

import os
import sys
import random
import time
from typing import Dict, List, Optional

# 添加路径
pyhanabi = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyhanabi)

from game_state import HleGameState
from agent import SimpleR3D2Agent, PiklAgent


class LocalHanabiGame:
    """本地Hanabi游戏类"""
    
    def __init__(self, human_name: str = "Human", bot_name: str = "Bot"):
        self.human_name = human_name
        self.bot_name = bot_name
        self.players = [human_name, bot_name]
        self.human_index = 0
        self.bot_index = 1
        
        # 初始化游戏状态
        self.game_state = None
        self.bot_agent = None
        self.bot_hidden_state = None
        
        # 游戏统计
        self.game_count = 0
        self.total_score = 0
        self.scores = []
        
    def load_bot(self, model_path: str, bot_type: str = "simple"):
        """加载AI模型"""
        print(f"🤖 正在加载AI模型: {model_path}")
        
        try:
            if bot_type == "simple":
                self.bot_agent = SimpleR3D2Agent(model_path)
            elif bot_type == "pikl":
                self.bot_agent = PiklAgent(model_path)
            else:
                raise ValueError(f"未知的bot类型: {bot_type}")
            
            print(f"✅ AI模型加载成功!")
            return True
        except Exception as e:
            print(f"❌ AI模型加载失败: {e}")
            return False
    
    def start_new_game(self):
        """开始新游戏"""
        print(f"\n🎮 开始新游戏 (第 {self.game_count + 1} 局)")
        print("="*50)
        
        # 创建游戏状态
        start_player = random.randint(0, 1)  # 随机选择先手
        self.game_state = HleGameState(
            players=self.players,
            my_name=self.human_name,
            start_player=start_player,
            hide_action=False,
            verbose=False
        )
        
        # 初始化AI的隐藏状态
        self.bot_hidden_state = self.bot_agent.init_and_get_h0(self.game_state)
        
        # 发牌
        self.deal_initial_cards()
        
        # 显示游戏开始信息
        starter_name = self.players[start_player]
        print(f"👥 玩家: {', '.join(self.players)}")
        print(f"🎯 先手: {starter_name}")
        print(f"🙋 你是玩家 {self.human_index} ({self.human_name})")
        
        # 开始游戏循环
        self.game_loop()
    
    def deal_initial_cards(self):
        """发初始手牌"""
        print(f"🔍 开始发初始手牌...")
        
        # 使用HanabiState的方式处理初始发牌
        # 在游戏开始时，通常会有一系列的机会事件（发牌）
        card_count = 0
        
        while (self.game_state.hle_state.cur_player() == -1 and  # CHANCE_PLAYER_ID
               not self.game_state.hle_state.is_terminal() and
               card_count < 10):  # 2个玩家 * 5张牌 = 10张
            
            if self.game_state.hle_state.deck_size() > 0:
                try:
                    print(f"🔍 发第 {card_count + 1} 张牌")
                    self.game_state.hle_state.deal_random_card()
                    card_count += 1
                except Exception as e:
                    print(f"❌ 发牌失败: {e}")
                    break
            else:
                print(f"🔍 牌堆已空，停止发牌")
                break
        
        print(f"🔍 发牌完成，共发了 {card_count} 张牌")
        
        # 显示发牌后的状态
        player_hands = self.game_state.hle_state.player_hands()
        for i, hand in enumerate(player_hands):
            print(f"🔍 玩家{i} ({self.players[i]}) 手牌数量: {len(hand)}")
    
    def game_loop(self):
        """主游戏循环"""
        turn_count = 0
        max_turns = 100  # 防止无限循环
        
        print(f"🔍 开始游戏循环...")
        
        while not self.game_state.hle_state.is_terminal() and turn_count < max_turns:
            current_player = self.game_state.hle_state.cur_player()
            
            # 处理机会事件（如果有的话）
            if current_player == -1:  # CHANCE_PLAYER_ID
                print(f"🔍 处理机会事件...")
                self.handle_chance_events()
                continue
            
            current_name = self.players[current_player]
            
            print(f"\n🎯 第 {turn_count + 1} 回合: {current_name} 的回合")
            print(f"🔍 当前玩家ID: {current_player}")
            
            # 检查游戏状态
            if self.game_state.hle_state.is_terminal():
                print(f"🔍 游戏已结束")
                break
            
            if current_player == self.human_index:
                # 人类玩家回合
                self.human_turn()
            elif current_player == self.bot_index:
                # AI回合
                self.bot_turn()
            else:
                print(f"⚠️  未知玩家ID: {current_player}")
                break
            
            turn_count += 1
            
            # 检查是否超过最大回合数
            if turn_count >= max_turns:
                print(f"⚠️  达到最大回合数限制 ({max_turns})，强制结束游戏")
                break
            
            time.sleep(0.5)  # 短暂暂停，便于观察
        
        print(f"🔍 游戏循环结束，总回合数: {turn_count}")
        
        # 游戏结束
        self.end_game()
    
    def human_turn(self):
        """人类玩家回合"""
        print("\n" + "="*50)
        print("🎯 轮到你了!")
        
        # 显示游戏状态
        self.display_game_state()
        
        # 获取合法动作
        legal_moves = self.get_legal_moves()
        
        if not legal_moves:
            print("⚠️  没有合法动作可执行")
            return
        
        # 显示动作选项
        print(f"\n🎯 可选动作:")
        for i, move in enumerate(legal_moves):
            print(f"   {i}: {move['description']}")
        
        # 获取玩家选择
        while True:
            try:
                choice = input(f"\n请选择动作 (0-{len(legal_moves)-1}): ")
                choice_idx = int(choice)
                
                if 0 <= choice_idx < len(legal_moves):
                    selected_move = legal_moves[choice_idx]
                    print(f"✅ 你选择了: {selected_move['description']}")
                    
                    # 执行动作
                    self.execute_move(selected_move['hle_move'])
                    break
                else:
                    print(f"❌ 无效选择，请输入 0-{len(legal_moves)-1}")
            except ValueError:
                print("❌ 请输入数字")
            except KeyboardInterrupt:
                print("\n👋 游戏结束!")
                sys.exit(0)
    
    def bot_turn(self):
        """AI回合"""
        print(f"\n🤖 {self.bot_name} 正在思考...")
        
        # 添加调试信息
        current_player = self.game_state.hle_state.cur_player()
        print(f"🔍 调试信息: 当前玩家 = {current_player}, 期望玩家 = {self.bot_index}")
        
        # 获取合法动作用于验证
        legal_moves = self.game_state.hle_state.legal_moves()
        print(f"🔍 合法动作数量: {len(legal_moves)}")
        
        # 获取AI的动作
        try:
            move, new_hidden_state = self.bot_agent.observe_and_maybe_act(
                self.game_state, self.bot_hidden_state
            )
            
            self.bot_hidden_state = new_hidden_state
            
            if move is not None:
                # 验证动作是否合法
                is_legal = self.game_state.hle_state.move_is_legal(move)
                print(f"🔍 AI选择的动作: {move}")
                print(f"🔍 动作是否合法: {is_legal}")
                
                if not is_legal:
                    print(f"⚠️  AI选择了不合法的动作，尝试选择第一个合法动作")
                    if legal_moves:
                        move = legal_moves[0]
                        print(f"🔍 替换为合法动作: {move}")
                    else:
                        print(f"❌ 没有合法动作可选择")
                        return
                
                # 显示AI的动作
                move_desc = self.describe_move(move)
                print(f"🤖 {self.bot_name} 选择: {move_desc}")
                
                # 执行动作
                self.execute_move(move)
            else:
                print(f"🤖 {self.bot_name} 无法执行动作")
                
        except Exception as e:
            print(f"❌ AI执行过程中出错: {e}")
            print(f"🔍 尝试使用第一个合法动作作为备选")
            if legal_moves:
                move = legal_moves[0]
                move_desc = self.describe_move(move)
                print(f"🤖 {self.bot_name} 备选动作: {move_desc}")
                self.execute_move(move)
            else:
                print(f"❌ 没有合法动作可选择")
    
    def get_legal_moves(self):
        """获取当前玩家的合法动作"""
        # 直接从 HanabiState 获取合法动作
        legal_moves_hle = self.game_state.hle_state.legal_moves()
        
        legal_moves = []
        
        for move in legal_moves_hle:
            move_desc = self.describe_move(move)
            legal_moves.append({
                'hle_move': move,
                'description': move_desc
            })
        
        return legal_moves
    
    def describe_move(self, hle_move):
        """描述HLE动作"""
        import hanalearn as hle
        
        move_type = hle_move.move_type()
        
        if move_type == hle.MoveType.Play:
            card_idx = hle_move.card_index()
            return f"打出手牌 {card_idx}"
        
        elif move_type == hle.MoveType.Discard:
            card_idx = hle_move.card_index()
            return f"丢弃手牌 {card_idx}"
        
        elif move_type == hle.MoveType.RevealColor:
            target_offset = hle_move.target_offset()
            target_idx = (self.human_index + target_offset) % len(self.players)
            target_name = self.players[target_idx]
            color = hle_move.color()
            color_names = ['红', '黄', '绿', '蓝', '紫']
            return f"给 {target_name} 提示颜色 {color_names[color]}"
        
        elif move_type == hle.MoveType.RevealRank:
            target_offset = hle_move.target_offset()
            target_idx = (self.human_index + target_offset) % len(self.players)
            target_name = self.players[target_idx]
            rank = hle_move.rank() + 1  # HLE使用0-4，显示时用1-5
            return f"给 {target_name} 提示数字 {rank}"
        
        else:
            return f"未知动作类型: {move_type}"
    
    def execute_move(self, hle_move):
        """执行HLE动作"""
        import hanalearn as hle
        
        print(f"🔍 执行动作: {hle_move}")
        
        # 验证动作是否合法
        if not self.game_state.hle_state.move_is_legal(hle_move):
            print(f"❌ 尝试执行不合法的动作: {hle_move}")
            return
        
        move_type = hle_move.move_type()
        current_player = self.game_state.hle_state.cur_player()
        
        print(f"🔍 动作类型: {move_type}, 当前玩家: {current_player}")
        
        if move_type == hle.MoveType.Play:
            card_idx = hle_move.card_index()
            player_hands = self.game_state.hle_state.player_hands()
            hand = player_hands[current_player]
            
            print(f"🔍 打牌: 玩家{current_player} 打出手牌{card_idx}, 手牌数量: {len(hand)}")
            
            if card_idx < len(hand):
                card = hand[card_idx]
                success = self.would_play_succeed(card)
                
                print(f"🔍 卡牌: {card}, 是否成功: {success}")
                
                if success:
                    print(f"✅ 成功打出 {card}")
                else:
                    print(f"❌ 打出失败 {card} (失去一个生命)")
                
                # 应用动作到游戏状态
                try:
                    self.game_state.hle_state.apply_move(hle_move)
                    print(f"🔍 动作应用成功")
                except Exception as e:
                    print(f"❌ 动作应用失败: {e}")
                    return
                
                # 处理机会事件（发牌）
                self.handle_chance_events()
        
        elif move_type == hle.MoveType.Discard:
            card_idx = hle_move.card_index()
            player_hands = self.game_state.hle_state.player_hands()
            hand = player_hands[current_player]
            
            print(f"🔍 弃牌: 玩家{current_player} 弃掉手牌{card_idx}, 手牌数量: {len(hand)}")
            
            if card_idx < len(hand):
                card = hand[card_idx]
                print(f"🔍 卡牌: {card}")
                print(f"🗑️  丢弃 {card} (获得一个提示)")
                
                # 应用动作到游戏状态
                try:
                    self.game_state.hle_state.apply_move(hle_move)
                    print(f"🔍 动作应用成功")
                except Exception as e:
                    print(f"❌ 动作应用失败: {e}")
                    return
                
                # 处理机会事件（发牌）
                self.handle_chance_events()
        
        elif move_type in [hle.MoveType.RevealColor, hle.MoveType.RevealRank]:
            target_offset = hle_move.target_offset()
            target_idx = (current_player + target_offset) % len(self.players)
            target_name = self.players[target_idx]
            
            print(f"🔍 提示: 玩家{current_player} 给玩家{target_idx}({target_name}) 提示")
            
            if move_type == hle.MoveType.RevealColor:
                color = hle_move.color()
                color_names = ['红', '黄', '绿', '蓝', '紫']
                print(f"💡 给 {target_name} 提示颜色 {color_names[color]}")
            else:
                rank = hle_move.rank() + 1
                print(f"💡 给 {target_name} 提示数字 {rank}")
            
            # 应用动作到游戏状态
            try:
                self.game_state.hle_state.apply_move(hle_move)
                print(f"🔍 动作应用成功")
            except Exception as e:
                print(f"❌ 动作应用失败: {e}")
                return
        
        print(f"🔍 动作执行完成")
    
    def would_play_succeed(self, card):
        """判断打出卡牌是否会成功"""
        # 使用 HanabiState 的 card_playable_on_fireworks 方法
        return self.game_state.hle_state.card_playable_on_fireworks(card.color, card.rank)
    
    def display_game_state(self):
        """显示游戏状态"""
        state = self.game_state.hle_state
        
        print(f"\n📊 游戏状态:")
        print(f"   得分: {state.score()}/25")
        print(f"   ❤️  生命: {state.life_tokens()}")
        print(f"   💡 提示: {state.information_tokens()}")
        print(f"   🃏 牌堆: {state.deck_size()}")
        
        # 显示烟花
        fireworks = state.fireworks()
        color_names = ['红', '黄', '绿', '蓝', '紫']
        fireworks_str = " ".join([f"{color_names[i]}:{fireworks[i]}" 
                                 for i in range(len(fireworks))])
        print(f"   🎆 烟花: {fireworks_str}")
        
        # 显示其他玩家的手牌（你能看到的）
        print(f"\n👀 对手手牌:")
        player_hands = state.player_hands()
        bot_hand = player_hands[self.bot_index]
        cards_str = " ".join([f"{i}:{card}" for i, card in enumerate(bot_hand)])
        print(f"   {self.bot_name}: {cards_str}")
        
        # 显示自己的手牌（你不知道具体内容）
        my_hand = player_hands[self.human_index]
        print(f"\n🃏 你的手牌 (你不知道具体内容):")
        cards_str = " ".join([f"{i}:??" for i in range(len(my_hand))])
        print(f"   {self.human_name}: {cards_str}")
        
        # 显示弃牌堆
        discards = state.discard_pile()
        if discards:
            print(f"\n🗑️  弃牌堆: {', '.join([str(card) for card in discards[-5:]])}")  # 显示最后5张
    
    def end_game(self):
        """游戏结束"""
        final_score = self.game_state.get_score()
        
        print(f"\n🎉 游戏结束!")
        print(f"🏆 最终得分: {final_score}/25")
        
        # 评价得分
        if final_score == 25:
            print("🌟 完美游戏! 🌟")
        elif final_score >= 20:
            print("🎊 出色的表现! 🎊")
        elif final_score >= 15:
            print("👍 不错的游戏! 👍")
        else:
            print("💪 下次会更好! 💪")
        
        # 更新统计
        self.game_count += 1
        self.total_score += final_score
        self.scores.append(final_score)
        
        # 显示统计信息
        if self.game_count > 1:
            avg_score = self.total_score / self.game_count
            print(f"\n📊 游戏统计:")
            print(f"   总局数: {self.game_count}")
            print(f"   平均得分: {avg_score:.2f}")
            print(f"   最高得分: {max(self.scores)}")
    
    def play_multiple_games(self, num_games: int = 1):
        """连续进行多局游戏"""
        for i in range(num_games):
            self.start_new_game()
            
            if i < num_games - 1:
                # 询问是否继续
                while True:
                    continue_game = input(f"\n是否继续下一局游戏? (y/n): ").lower()
                    if continue_game in ['y', 'yes', '是']:
                        break
                    elif continue_game in ['n', 'no', '否']:
                        print("👋 感谢游戏!")
                        return
                    else:
                        print("请输入 y 或 n")

    def handle_chance_events(self):
        """处理机会事件（主要是发牌）"""
        # 检查是否需要处理机会事件
        while (self.game_state.hle_state.cur_player() == -1 and  # CHANCE_PLAYER_ID
               not self.game_state.hle_state.is_terminal()):
            
            print(f"🔍 处理机会事件: 牌堆剩余 {self.game_state.hle_state.deck_size()} 张")
            
            if self.game_state.hle_state.deck_size() > 0:
                try:
                    self.game_state.hle_state.deal_random_card()
                    print(f"🔍 发牌成功")
                except Exception as e:
                    print(f"❌ 发牌失败: {e}")
                    break
            else:
                print(f"🔍 牌堆已空，无法发牌")
                break


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="本地Hanabi对战程序")
    parser.add_argument("--model", required=True, help="AI模型路径")
    parser.add_argument("--bot_type", default="simple", choices=["simple", "pikl"], 
                       help="AI模型类型")
    parser.add_argument("--human_name", default="Human", help="人类玩家名称")
    parser.add_argument("--bot_name", default="Bot", help="AI玩家名称")
    parser.add_argument("--games", type=int, default=1, help="游戏局数")
    
    args = parser.parse_args()
    
    print("🎮 本地Hanabi对战程序")
    print("="*50)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 创建游戏实例
    game = LocalHanabiGame(args.human_name, args.bot_name)
    
    # 加载AI模型
    if not game.load_bot(args.model, args.bot_type):
        sys.exit(1)
    
    print(f"\n🎯 游戏规则:")
    print("   - 目标: 合作完成烟花，最高得分25分")
    print("   - 你看不到自己的手牌，但能看到对手的")
    print("   - 可以打出卡牌、丢弃卡牌或给对手提示")
    print("   - 打出错误卡牌会失去生命，生命为0游戏结束")
    print("   - 提示消耗提示令牌，丢弃卡牌可获得提示令牌")
    
    try:
        # 开始游戏
        game.play_multiple_games(args.games)
    except KeyboardInterrupt:
        print("\n👋 游戏被中断!")
    
    print("\n感谢使用本地Hanabi对战程序!")


if __name__ == "__main__":
    # 测试正确的游戏初始化
    print("🔍 测试游戏初始化...")
    
    # 创建游戏状态
    game_state = HleGameState(
        players=["Human", "Bot"],
        my_name="Human",
        start_player=0,  # Human先手
        hide_action=False,
        verbose=True
    )
    obs,_,_ = game_state.observe_readable()
    
    print(f"🔍 初始游戏状态:")
    print(f"   牌堆大小: {obs.deck_size()}")

    
    # 手动发牌 - 每人5张牌
    print(f"\n🔍 开始发牌...")
    card_order = 0
    
    for round_num in range(5):  # 5轮发牌
        for player_idx in range(2):  # 2个玩家
            # 生成随机卡牌
            color = random.randint(0, 4)  # 0-4 对应 5种颜色
            rank = random.randint(1, 5)   # 1-5 对应 5个等级
            
            print(f"🔍 给玩家{player_idx} 发牌: 颜色={color}, 等级={rank}, 顺序={card_order}")
            
            # 调用draw方法发牌
            game_state.draw(player_idx, color, rank, card_order)
            card_order += 1
    obs,_,_ = game_state.observe_readable()
    print(f"\n🔍 发牌完成后的游戏状态:")
    print(f"   牌堆大小: {obs.deck_size()}")

    
    # 显示手牌
    print(f"\n🔍 玩家手牌:")
    for i, hand in enumerate(game_state.hands):
        print(f"   玩家{i}: {len(hand.cards)} 张牌")
        for j, card in enumerate(hand.cards):
            print(f"     {j}: {card} (order={card.order})")
    
    # # 测试观察
    # print(f"\n🔍 测试观察功能:")
    obs, legal_moves, last_move = game_state.observe_readable()
    print(f"   合法动作数量: {len(legal_moves)}")
    # print(f"   提示令牌: {obs.information_tokens()}")
    # print(f"   生命令牌: {obs.life_tokens()}")
    # print(f"   得分: {obs.score()}")

    # 显示合法动作  
    print(f"\n🔍 合法动作:")
    for i, move in enumerate(legal_moves):
        print(f"   {i}: {move.to_string()}")
    
    #main() 