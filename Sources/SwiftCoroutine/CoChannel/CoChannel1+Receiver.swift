//
//  CoChannel1+Receiver.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoChannel {
    
    /// A `CoChannel` wrapper that provides receive-only functionality.
    public class Receiver {
        
        /// The maximum number of elements that can be stored in a channel.
        public var maxBufferSize: Int { 0 }
        
        /// Returns a number of elements in this channel.
        public var count: Int { 0 }
        
        /// Returns `true` if the channel is empty (contains no elements), which means no elements to receive.
        public var isEmpty: Bool { true }
        
        // MARK: - receive
        
        /// Retrieves and removes an element from this channel if it’s not empty, or suspends a coroutine while the channel is empty.
        /// - Throws: CoChannelError when canceled or closed.
        /// - Returns: Removed value from the channel.
        public func awaitReceive() throws -> Element {
            throw CoChannelError.canceled
        }

        /// Creates `CoFuture` with retrieved value from this channel.
        /// - Returns: `CoFuture` with a future value from the channel.
        public func receiveFuture() -> CoFuture<Element> {
            CoFuture(_result: .failure(CoFutureError.canceled))
        }
        
        /// Retrieves and removes an element from this channel.
        /// - Returns: Element from this channel if its not empty, or returns nill if the channel is empty or is closed or canceled.
        public func poll() -> Element? { nil }

        /// Adds an observer callback to receive an element from this channel.
        /// - Parameter callback: The callback that is called when a value is received.
        public func whenReceive(_ callback: @escaping (Result<Element, CoChannelError>) -> Void) {
            callback(.failure(.canceled))
        }
        
        // MARK: - map
        
        /// Returns new `Receiver` that provides transformed values from this `Receiver`.
        /// - Parameter transform: A mapping closure.
        /// - returns: A `Receiver` with transformed values.
        public final func map<T>(_ transform: @escaping (Element) -> T) -> CoChannel<T>.Receiver {
            CoChannelMap(receiver: self, transform: transform)
        }
        
        // MARK: - close
        
        /// Returns `true` if the channel is closed.
        public var isClosed: Bool { true }
        
        // MARK: - cancel
        
        /// Closes the channel and removes all buffered sent elements from it.
        public func cancel() {}
        
        /// Returns `true` if the channel is canceled.
        public var isCanceled: Bool { true }
        
        /// Adds an observer callback that is called when the `CoChannel` is completed (closed, canceled or deinited).
        /// - Parameter callback: The callback that is called when the `CoChannel` is completed.
        public func whenComplete(_ callback: @escaping () -> Void) {
            callback()
        }
        
        // MARK: - sequence
        
        /// Make an iterator which successively retrieves and removes values from the channel.
        ///
        /// If `next()` was called inside a coroutine and there are no more elements in the channel,
        /// then the coroutine will be suspended until a new element will be added to the channel or it will be closed or canceled.
        /// - Returns: Iterator for the channel elements.
        final public func makeIterator() -> AnyIterator<Element> {
            AnyIterator { Coroutine.isInsideCoroutine ? try? self.awaitReceive() : self.poll() }
        }
        
    }
    
}
