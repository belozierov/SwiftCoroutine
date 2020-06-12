//
//  CoChannelSubscription.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 11.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import Combine

@available(OSX 10.15, iOS 13.0, *)
internal final class CoChannelSubscription<S: Subscriber, T>: Subscription where S.Input == T, S.Failure == CoChannelError {
    
    private let receiver: CoChannel<T>.Receiver
    private var subscriber: S?
    
    @inlinable internal init(subscriber: S, receiver: CoChannel<T>.Receiver) {
        self.receiver = receiver
        self.subscriber = subscriber
        @inline(__always) func subscribe() {
            receiver.whenReceive { result in
                guard let subscriber = self.subscriber else { return }
                switch result {
                case .success(let result):
                    _ = subscriber.receive(result)
                    subscribe()
                case .failure(let error) where error == .canceled:
                    subscriber.receive(completion: .failure(error))
                case .failure:
                    subscriber.receive(completion: .finished)
                }
            }
        }
        subscribe()
    }
    
    @inlinable internal func cancel() {
        subscriber = nil
    }
    
    @inlinable internal func request(_ demand: Subscribers.Demand) {}
    
}
#endif
