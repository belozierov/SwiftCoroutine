//
//  CoChannelSender.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoChannel {
    
    /// A `CoChannel` wrapper that provides send-only functionality.
    public final class Sender {
        
        @usableFromInline internal let channel: _Channel<Element>
        
        @usableFromInline internal init(channel: _Channel<Element>) {
            self.channel = channel
        }
        
        /// The type of channel buffer.
        @inlinable public var bufferType: BufferType {
            channel.bufferType
        }
        
    }
    
}

extension CoChannel.Sender {
    
    /// Returns a number of elements in this channel.
    @inlinable public var count: Int {
        channel.count
    }
    
    /// Returns `true` if the channel is empty (contains no elements), which means no elements to receive.
    @inlinable public var isEmpty: Bool {
        channel.isEmpty
    }
    
    // MARK: - send
    
    /// Sends the element to this channel, suspending the coroutine while the buffer of this channel is full. Must be called inside a coroutine.
    /// - Parameter element: Value that will be sent to the channel.
    /// - Throws: CoChannelError when canceled or closed.
    @inlinable public func awaitSend(_ element: Element) throws {
        try channel.awaitSend(element)
    }
    
    /// Adds the future's value to this channel when it will be available.
    /// - Parameter future: `CoFuture`'s value that will be sent to the channel.
    @inlinable public func sendFuture(_ future: CoFuture<Element>) {
        channel.sendFuture(future)
    }
    
    /// Immediately adds the value to this channel, if this doesn’t violate its capacity restrictions, and returns true.
    /// Otherwise, just returns false.
    /// - Parameter element: Value that might be sent to the channel.
    /// - Returns:`true` if sent successfully or `false` if channel buffer is full or channel is closed or canceled.
    @discardableResult @inlinable public func offer(_ element: Element) -> Bool {
        channel.offer(element)
    }
    
    // MARK: - close
    
    /// Closes this channel. No more send should be performed on the channel.
    /// - Returns: `true` if closed successfully or `false` if channel is already closed or canceled.
    @discardableResult @inlinable public func close() -> Bool {
        channel.close()
    }
    
    /// Returns `true` if the channel is closed.
    @inlinable public var isClosed: Bool {
        channel.isClosed
    }
    
    // MARK: - cancel
    
    /// Closes the channel and removes all buffered sent elements from it.
    @inlinable public func cancel() {
        channel.cancel()
    }
    
    /// Returns `true` if the channel is canceled.
    @inlinable public var isCanceled: Bool {
        channel.isCanceled
    }
    
    /// Adds an observer callback that is called when the `CoChannel` is canceled.
    /// - Parameter callback: The callback that is called when the `CoChannel` is canceled.
    public func whenCanceled(_ callback: @escaping () -> Void) {
        channel.whenCanceled(callback)
    }
    
    // MARK: - complete
    
    /// Adds an observer callback that is called when the `CoChannel` is completed (closed, canceled or deinited).
    /// - Parameter callback: The callback that is called when the `CoChannel` is completed.
    @inlinable public func whenComplete(_ callback: @escaping () -> Void) {
        channel.whenComplete(callback)
    }
    
}
