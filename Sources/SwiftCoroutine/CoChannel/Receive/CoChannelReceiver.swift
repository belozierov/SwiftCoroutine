//
//  CoChannelReceiver.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal protocol CoChannelReceiver {
    
    associatedtype Output
    func awaitReceive() throws -> Output
    func poll() -> Output?
    func receiveFuture() -> CoFuture<Output>
    func whenReceive(_ callback: @escaping (Result<Output, CoChannelError>) -> Void)
    var count: Int { get }
    var isEmpty: Bool { get }
    var isClosed: Bool { get }
    func cancel()
    var isCanceled: Bool { get }
    func whenCanceled(_ callback: @escaping () -> Void)
    var maxBufferSize: Int { get }
    
}

extension CoChannel: CoChannelReceiver {}
extension CoChannel.Receiver: CoChannelReceiver {}
