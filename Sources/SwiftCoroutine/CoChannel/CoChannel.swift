//
//  CoChannel.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 02.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

/// Channel is a non-blocking primitive for communication between a sender and a receiver.
/// Conceptually, a channel is similar to a queue that allows to suspend a coroutine on receive if it is empty or on send if it is full.
///
/// - Important: Always `close()` or `cancel()` a channel when you are done to resume all suspended coroutines by the channel.
///
/// ```
/// let channel = CoChannel<Int>(capacity: 1)
///
/// DispatchQueue.global().startCoroutine {
///    for i in 0..<100 {
///        try channel.awaitSend(i)
///    }
///    channel.close()
/// }
///
/// DispatchQueue.global().startCoroutine {
///     for i in channel.makeIterator() {
///         print("Receive", i)
///     }
///     print("Done")
/// }
/// ```
public final class CoChannel<Element> {
    
    /// `CoChannel` buffer type.
    public enum BufferType: Equatable {
        /// This channel does not have any buffer.
        ///
        /// An element is transferred from the sender to the receiver only when send and receive invocations meet in time,
        /// so `awaitSend(_:)` suspends until invokes receive, and `awaitReceive()` suspends until invokes send.
        case none
        /// This channel have a buffer with the specified capacity.
        ///
        /// `awaitSend(_:)` suspends only when the buffer is full,
        /// and `awaitReceive()` suspends only when the buffer is empty.
        case buffered(capacity: Int)
        /// This channel has a buffer with unlimited capacity.
        ///
        /// `awaitSend(_:)` to this channel never suspends, and offer always returns true.
        /// `awaitReceive()` suspends only when the buffer is empty.
        case unlimited
        /// This channel buffers at most one element and offer invocations,
        /// so that the receiver always gets the last element sent.
        ///
        /// Only the last sent element is received, while previously sent elements are lost.
        /// `awaitSend(_:)` to this channel never suspends, and offer always returns true.
        /// `awaitReceive()` suspends only when the buffer is empty.
        case conflated
    }
    
    @usableFromInline internal let channel: _Channel<Element>
    
    /// Initializes a channel.
    /// - Parameter type: The type of channel buffer.
    public init(bufferType type: BufferType = .unlimited) {
        switch type {
        case .conflated:
            channel = _ConflatedChannel()
        case .buffered(let capacity):
            channel = _BufferedChannel(capacity: capacity)
        case .unlimited:
            channel = _BufferedChannel(capacity: .max)
        case .none:
            channel = _BufferedChannel(capacity: 0)
        }
    }
    
    /// Initializes a channel with `BufferType.buffered(capacity:)` .
    /// - Parameter capacity: The maximum number of elements that can be stored in a channel.
    public init(capacity: Int) {
        channel = _BufferedChannel(capacity: capacity)
    }
    
}

extension CoChannel {
    
    /// The type of channel buffer.
    @inlinable public var bufferType: BufferType {
        channel.bufferType
    }
    
    /// Returns tuple of `Receiver` and `Sender`.
    @inlinable public var pair: (receiver: Receiver, sender: Sender) {
        (channel, sender)
    }
    
    // MARK: - send
    
    /// A `CoChannel` wrapper that provides send-only functionality.
    @inlinable public var sender: Sender {
        Sender(channel: channel)
    }
    
    /// Sends the element to this channel, suspending the coroutine while the buffer of this channel is full.
    /// Must be called inside a coroutine.
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
    
    // MARK: - receive
    
    /// A `CoChannel` wrapper that provides receive-only functionality.
    @inlinable public var receiver: Receiver { channel }
    
    /// Retrieves and removes an element from this channel if it’s not empty, or suspends a coroutine while the channel is empty.
    /// - Throws: CoChannelError when canceled or closed.
    /// - Returns: Removed value from the channel.
    @inlinable public func awaitReceive() throws -> Element {
        try channel.awaitReceive()
    }
    
    /// Creates `CoFuture` with retrieved value from this channel.
    /// - Returns: `CoFuture` with a future value from the channel.
    @inlinable public func receiveFuture() -> CoFuture<Element> {
        channel.receiveFuture()
    }
    
    /// Retrieves and removes an element from this channel.
    /// - Returns: Element from this channel if its not empty, or returns nill if the channel is empty or is closed or canceled.
    @inlinable public func poll() -> Element? {
        channel.poll()
    }
    
    /// Adds an observer callback to receive an element from this channel.
    /// - Parameter callback: The callback that is called when a value is received.
    @inlinable public func whenReceive(_ callback: @escaping (Result<Element, CoChannelError>) -> Void) {
        channel.whenReceive(callback)
    }
    
    /// Returns a number of elements in this channel.
    @inlinable public var count: Int {
        channel.count
    }
    
    /// Returns `true` if the channel is empty (contains no elements), which means no elements to receive.
    @inlinable public var isEmpty: Bool {
        channel.isEmpty
    }
    
    // MARK: - map
    
    /// Returns new `Receiver` that provides transformed values from this `CoChannel`.
    /// - Parameter transform: A mapping closure.
    /// - returns: A `Receiver` with transformed values.
    @inlinable public func map<T>(_ transform: @escaping (Element) -> T) -> CoChannel<T>.Receiver {
        channel.map(transform)
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
    @inlinable public func whenCanceled(_ callback: @escaping () -> Void) {
        channel.whenCanceled(callback)
    }
    
    // MARK: - complete
    
    /// Adds an observer callback that is called when the `CoChannel` is completed (closed, canceled or deinited).
    /// - Parameter callback: The callback that is called when the `CoChannel` is completed.
    @inlinable public func whenComplete(_ callback: @escaping () -> Void) {
        channel.whenComplete(callback)
    }
    
}

extension CoChannel {
    
    // MARK: - sequence
    
    /// Make an iterator which successively retrieves and removes values from the channel.
    ///
    /// If `next()` was called inside a coroutine and there are no more elements in the channel,
    /// then the coroutine will be suspended until a new element will be added to the channel or it will be closed or canceled.
    /// - Returns: Iterator for the channel elements.
    @inlinable public func makeIterator() -> AnyIterator<Element> {
        channel.makeIterator()
    }
    
}
