//
//  CoChannelReceiverProtocol.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 11.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal protocol CoChannelReceiverProtocol {
    
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
    func whenComplete(_ callback: @escaping () -> Void)
    var maxBufferSize: Int { get }
    
}

extension CoChannel: CoChannelReceiverProtocol {}
extension CoChannel.Receiver: CoChannelReceiverProtocol {}
