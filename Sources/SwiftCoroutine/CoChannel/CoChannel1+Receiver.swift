//
//  CoChannel1+Receiver.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoChannel {
    
    /// A `CoChannel` wrapper that provides receive-only functionality.
    ///
    /// - Note: `Receiver` has reference semantics.
    ///
    public struct Receiver {
        
        internal enum Source {
            case channel(CoChannel)
            case wrapper(CoChannelReceiverWrapper<Element>)
        }
        
        internal let source: Source
        
    }
    
}

extension CoChannel.Receiver {
    
    /// The maximum number of elements that can be stored in a channel.
    public var maxBufferSize: Int {
        switch source {
        case .channel(let channel):
            return channel.maxBufferSize
        case .wrapper(let wrapper):
            return wrapper.maxBufferSize
        }
    }
    
    /// Returns a number of elements in this channel.
    public var count: Int {
        switch source {
        case .channel(let channel):
            return channel.count
        case .wrapper(let wrapper):
            return wrapper.count
        }
    }
    
    /// Returns `true` if the channel is empty (contains no elements), which means no elements to receive.
    public var isEmpty: Bool {
        switch source {
        case .channel(let channel):
            return channel.isEmpty
        case .wrapper(let wrapper):
            return wrapper.isEmpty
        }
    }
    
    // MARK: - receive
    
    /// Retrieves and removes an element from this channel if it’s not empty, or suspends a coroutine while the channel is empty.
    /// - Throws: CoChannelError when canceled or closed.
    /// - Returns: Removed value from the channel.
    public func awaitReceive() throws -> Element {
        switch source {
        case .channel(let channel):
            return try channel.awaitReceive()
        case .wrapper(let wrapper):
            return try wrapper.awaitReceive()
        }
    }

    /// Creates `CoFuture` with retrieved value from this channel.
    /// - Returns: `CoFuture` with a future value from the channel.
    public func receiveFuture() -> CoFuture<Element> {
        switch source {
        case .channel(let channel):
            return channel.receiveFuture()
        case .wrapper(let wrapper):
            return wrapper.receiveFuture()
        }
    }
    
    /// Retrieves and removes an element from this channel.
    /// - Returns: Element from this channel if its not empty, or returns nill if the channel is empty or is closed or canceled.
    public func poll() -> Element? {
        switch source {
        case .channel(let channel):
            return channel.poll()
        case .wrapper(let wrapper):
            return wrapper.poll()
        }
    }

    /// Adds an observer callback to receive an element from this channel.
    /// - Parameter callback: The callback that is called when a value is received.
    public func whenReceive(_ callback: @escaping (Result<Element, CoChannelError>) -> Void) {
        switch source {
        case .channel(let channel):
            return channel.whenReceive(callback)
        case .wrapper(let wrapper):
            return wrapper.whenReceive(callback)
        }
    }
    
    // MARK: - map
    
    /// Returns new `Receiver` that provides transformed values from this `Receiver`.
    /// - Parameter transform: A mapping closure.
    /// - returns: A `Receiver` with transformed values.
    public func map<T>(_ transform: @escaping (Element) -> T) -> CoChannel<T>.Receiver {
        let wrapper = CoChannelMap(receiver: self, transform: transform)
        return CoChannel<T>.Receiver(source: .wrapper(wrapper))
    }
    
    // MARK: - close
    
    /// Returns `true` if the channel is closed.
    public var isClosed: Bool {
        switch source {
        case .channel(let channel):
            return channel.isClosed
        case .wrapper(let wrapper):
            return wrapper.isClosed
        }
    }
    
    // MARK: - cancel
    
    /// Closes the channel and removes all buffered sent elements from it.
    public func cancel() {
        switch source {
        case .channel(let channel):
            return channel.cancel()
        case .wrapper(let wrapper):
            return wrapper.cancel()
        }
    }
    
    /// Returns `true` if the channel is canceled.
    public var isCanceled: Bool {
        switch source {
        case .channel(let channel):
            return channel.isCanceled
        case .wrapper(let wrapper):
            return wrapper.isCanceled
        }
    }
    
    /// Adds an observer callback that is called when the `CoChannel` is canceled.
    /// - Parameter callback: The callback that is called when the `CoChannel` is canceled.
    public func whenCanceled(_ callback: @escaping () -> Void) {
        switch source {
        case .channel(let channel):
            return channel.whenCanceled(callback)
        case .wrapper(let wrapper):
            return wrapper.whenCanceled(callback)
        }
    }
    
}

extension CoChannel.Receiver {
    
    // MARK: - sequence
    
    /// Make an iterator which successively retrieves and removes values from the channel.
    ///
    /// If `next()` was called inside a coroutine and there are no more elements in the channel,
    /// then the coroutine will be suspended until a new element will be added to the channel or it will be closed or canceled.
    /// - Returns: Iterator for the channel elements.
    public func makeIterator() -> AnyIterator<Element> {
        switch source {
        case .channel(let channel):
            return channel.makeIterator()
        case .wrapper(let wrapper):
            return AnyIterator {
                Coroutine.isInsideCoroutine
                    ? try? wrapper.awaitReceive()
                    : wrapper.poll()
            }
        }
    }
    
}
