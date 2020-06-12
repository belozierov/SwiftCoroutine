//
//  CoChannelPublisher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 11.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import Combine

@available(OSX 10.15, iOS 13.0, *)
internal final class CoChannelPublisher<Output> {
    
    internal typealias Failure = CoChannelError
    internal let receiver: CoChannel<Output>.Receiver
    
    @inlinable internal init(receiver: CoChannel<Output>.Receiver) {
        self.receiver = receiver
    }
    
}

@available(OSX 10.15, iOS 13.0, *)
extension CoChannelPublisher: Publisher {
    
    @inlinable internal func receive<S: Subscriber>(subscriber: S) where Failure == S.Failure, Output == S.Input {
        let subscription = CoChannelSubscription(subscriber: subscriber, receiver: receiver)
        subscriber.receive(subscription: subscription)
    }
    
}
#endif
