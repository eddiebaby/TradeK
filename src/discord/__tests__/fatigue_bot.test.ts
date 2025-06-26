import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { MockDiscordClient } from './__mocks__/discord-mock';
import { FatigueBot } from '../fatigue_bot';

/**
 * Mock implementation of discord.Client for testing
 */
class MockDiscordClient {
  mockMessage(content: string, channel: string) {
    return {
      content,
      channel: {
        id: channel,
        name: channel,
      },
      author: {
        id: 'test-user',
        username: 'Test User',
        permissions: {
          has: () => true,
        },
      },
      delete: jest.fn(),
      timeout: jest.fn(),
      channel: {
        send: jest.fn(),
      },
    };
  }
}

describe('Fatigue Bot', () => {
  let bot: FatigueBot;
  let mockClient: MockDiscordClient;

  beforeEach(() => {
    mockClient = new MockDiscordClient();
    bot = new FatigueBot(mockClient);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('onMessage', () => {
    it('should timeout user when message contains fatigue phrase', async () => {
      const message = mockClient.mockMessage(
        'I am tired',
        'general'
      );
      await bot.onMessage(message);
      expect(message.delete).toHaveBeenCalledWith();
      expect(message.author.timeout).toHaveBeenCalledWith(60000);
      expect(message.channel.send).toHaveBeenCalledWith(
        '⏰ Test user has been timed out for 1 minute for fatigue violation!'
      );
    });

    it('should not timeout moderator', async () => {
      const message = mockClient.mockMessage(
        'I am tired',
        'general'
      );
      message.author.permissions.has.mockReturnValue(true);
      await bot.onMessage(message);
      expect(message.delete).not.toHaveBeenCalled();
      expect(message.author.timeout).not.toHaveBeenCalled();
    });

    it('should not timeout bot itself', async () => {
      const message = mockClient.mockMessage(
        'I am tired',
        'general'
      );
      message.author.id = bot.user!.id;
      await bot.onMessage(message);
      expect(message.delete).not.toHaveBeenCalled();
      expect(message.author.timeout).not.toHaveBeenCalled();
    });

    it('should handle permission errors', async () => {
      const message = mockClient.mockMessage(
        'I am tired',
        'general'
      );
      message.author.permissions.has.mockReturnValue(false);
      await bot.onMessage(message);
      expect(message.delete).not.toHaveBeenCalled();
      expect(message.author.timeout).not.toHaveBeenCalled();
    });

    it('should handle message deletion errors', async () => {
      const message = mockClient.mockMessage(
        'I am tired',
        'general'
      );
      message.delete.mockRejectedValue(new Error('Delete failed'));
      await bot.onMessage(message);
      expect(message.channel.send).toHaveBeenCalledWith(
        'Error: Failed to delete message'
      );
    });

    it('should handle timeout errors', async () => {
      const message = mockClient.mockMessage(
        'I am tired',
        'general'
      );
      message.author.timeout.mockRejectedValue(new Error('Timeout failed'));
      await bot.onMessage(message);
      expect(message.channel.send).toHaveBeenCalledWith(
        'Error: Failed to apply timeout'
      );
    });
  });
});