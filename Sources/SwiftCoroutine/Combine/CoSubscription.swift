//
//  CoSubscription.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 26.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import Combine

@available(OSX 10.15, iOS 13.0, *)
class CoSubscription: Subscription {
    
    typealias Canceller = (CoSubscription) -> Void
    private let canceller: Canceller
    
    init(canceller: @escaping Canceller) {
        self.canceller = canceller
    }
    
    @inlinable func cancel() {
        canceller(self)
    }
    
    @inlinable func request(_ demand: Subscribers.Demand) {}
    
}

@available(OSX 10.15, iOS 13.0, *)
extension CoSubscription: Hashable {
    
    static func == (lhs: CoSubscription, rhs: CoSubscription) -> Bool {
        lhs === rhs
    }
    
    func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
#endif
